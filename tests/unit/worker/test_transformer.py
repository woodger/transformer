import pytest
import torch

from app.contracts.semantic.v5 import ModelContract
from app.worker.model.context import (
    context_input_dim,
    context_key_padding_mask,
    context_missingness_ratios,
    context_token_ratios,
    prepare_context_input,
)
from app.worker.model.transformer import TransformerModel, public_predictions
from tests.fixture_documents import semantic_fixture_document


def _contract(
    *,
    mode: str = "relaxed",
    normalization_order: str = "postNorm",
    hidden: int = 32,
    nhead: int = 4,
):
    document = semantic_fixture_document("multi-target-shared-resource")[
        "modelContract"
    ]
    assert isinstance(document, dict)
    tuning = document["modelTuning"]
    assert isinstance(tuning, dict)
    tuning.update({
        "hiddenWidth": hidden,
        "encoderLayerCount": 1,
        "dropoutProbability": 0.0,
        "attentionHeadCount": nhead,
        "missingValuePolicy": mode,
        "encoderNormalizationOrder": normalization_order,
    })
    return ModelContract.from_document(document)


def test_transformer_forward_shape():
    batch = 4
    seq_len = 10
    feat_dim = 8
    contract = _contract()

    model = TransformerModel(
        input_dim=feat_dim,
        seq_len=seq_len,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
        normalization_order="postNorm",
    )

    features = torch.randn(batch, seq_len, feat_dim)
    model_output = model(features)

    assert model_output.shape == (batch, 4)
    assert public_predictions(
        model_output,
        contract,
    ).shape == (batch, 3)


@pytest.mark.parametrize("hidden", [1, 3, 9])
def test_transformer_supports_odd_hidden_width(hidden):
    contract = _contract(hidden=hidden, nhead=1)
    model = TransformerModel(
        input_dim=2,
        seq_len=3,
        hidden_dim=hidden,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=1,
        normalization_order="postNorm",
    )

    predictions = public_predictions(model(torch.randn(2, 3, 2)), contract)

    assert predictions.shape == (2, contract.target_width)
    assert torch.isfinite(predictions).all()


def test_transformer_supports_sequences_beyond_default_positional_capacity():
    contract = _contract(hidden=8, nhead=1)
    model = TransformerModel(
        input_dim=1,
        seq_len=5001,
        hidden_dim=8,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=1,
        normalization_order="postNorm",
    ).eval()

    with torch.no_grad():
        predictions = public_predictions(model(torch.zeros(1, 5001, 1)), contract)

    assert predictions.shape == (1, contract.target_width)
    assert torch.isfinite(predictions).all()


def test_transformer_keeps_existing_checkpoint_positional_buffer_shape():
    model = TransformerModel(
        input_dim=1,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=_contract(),
        nhead=4,
        normalization_order="postNorm",
    )

    assert model.state_dict()["pos.pe"].shape == (5000, 32)


@pytest.mark.parametrize(
    ("normalization_order", "norm_first"),
    (("postNorm", False), ("preNorm", True)),
)
def test_transformer_materializes_declared_encoder_normalization_order(
    normalization_order: str,
    norm_first: bool,
):
    model = TransformerModel(
        input_dim=8,
        seq_len=3,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
        model_contract=_contract(normalization_order=normalization_order),
        nhead=4,
        normalization_order=normalization_order,
    )

    assert all(layer.norm_first is norm_first for layer in model.encoder.layers)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (torch.zeros(2, 24), "shape"),
        (torch.zeros(2, 2, 8), "sequence length"),
        (torch.zeros(2, 3, 7), "feature dimension"),
        (torch.zeros(2, 3, 8, dtype=torch.float64), "float32"),
    ],
)
def test_transformer_rejects_input_outside_tensor_contract(features, message):
    contract = _contract()
    model = TransformerModel(
        input_dim=8,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
        normalization_order="postNorm",
    )

    with pytest.raises(ValueError, match=message):
        model(features)


def test_transformer_rejects_invalid_attention_dimensions():
    contract = _contract()
    with pytest.raises(ValueError, match="divisible"):
        TransformerModel(
            input_dim=8,
            seq_len=3,
            hidden_dim=30,
            layers=1,
            dropout=0.0,
            model_contract=contract,
            nhead=8,
            normalization_order="postNorm",
        )


def test_transformer_input_dim_matches_context_mode():
    relaxed_contract = _contract()
    strict_contract = _contract(
        mode="strict",
    )
    relaxed = TransformerModel(
        input_dim=8,
        seq_len=10,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=relaxed_contract,
        nhead=4,
        context_mode="relaxed",
        normalization_order="postNorm",
    )
    strict = TransformerModel(
        input_dim=8,
        seq_len=10,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=strict_contract,
        nhead=4,
        context_mode="strict",
        normalization_order="postNorm",
    )

    assert context_input_dim(8, "relaxed") == 16
    assert relaxed.input_proj.in_features == 16
    assert strict.input_proj.in_features == 8


def test_context_mask_depends_on_context_mode():
    features = torch.tensor([
        [
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    assert context_key_padding_mask(features, "strict").tolist() == [[True, True]]
    assert context_key_padding_mask(features, "relaxed").tolist() == [[False, True]]


def test_context_token_ratios_split_complete_partial_and_empty_tokens():
    features = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
            [4.0, 5.0, 6.0],
        ],
    ])

    ratios = context_token_ratios(features)

    assert ratios["masked_token_ratio"] == 0.25
    assert ratios["complete_token_ratio"] == 0.5
    assert ratios["partial_token_ratio"] == 0.25
    assert ratios["empty_token_ratio"] == 0.25


def test_context_token_ratios_use_selected_masking_mode():
    features = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    assert context_token_ratios(features, "strict")["masked_token_ratio"] == pytest.approx(2 / 3)
    assert context_token_ratios(features, "relaxed")["masked_token_ratio"] == pytest.approx(1 / 3)


def test_context_missingness_ratios_include_values_and_tokens():
    features = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    ratios = context_missingness_ratios(features, "relaxed")

    assert ratios == pytest.approx({
        "nan_ratio": 4 / 9,
        "masked_token_ratio": 1 / 3,
        "complete_token_ratio": 1 / 3,
        "partial_token_ratio": 1 / 3,
        "empty_token_ratio": 1 / 3,
    })


def test_relaxed_keeps_missing_flags_after_nan_to_num():
    features = torch.tensor([
        [
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    prepared_context = prepare_context_input(features, "relaxed")

    assert prepared_context.key_padding_mask.tolist() == [[False, True]]
    assert prepared_context.features.tolist() == [
        [
            [1.0, 0.0, 3.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
    ]


def test_transformer_forward_with_partial_and_full_nan_tokens_is_finite():
    contract = _contract()
    model = TransformerModel(
        input_dim=3,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
        context_mode="relaxed",
        normalization_order="postNorm",
    )

    features = torch.randn(2, 3, 3)
    features[0, 1, 2] = float("nan")
    features[0, 2, :] = float("nan")
    features[1, :, :] = float("nan")

    model_output = model(features)

    assert model_output.shape == (2, 4)
    assert torch.isfinite(model_output).all()


def test_public_predictions_apply_declared_slot_transformations():
    contract = _contract()
    output = torch.tensor([
        [-0.25, 0.0, 2.0, 3.5],
    ])

    prediction = public_predictions(output, contract)

    assert prediction.shape == (1, 3)
    assert prediction[0, 0] == pytest.approx(-0.25)
    assert prediction[0, 1] == pytest.approx(0.5)
    assert prediction[0, 2] == pytest.approx(
        torch.sigmoid(torch.tensor(2.0)).item()
    )
    assert torch.all((prediction[:, 1:] >= 0) & (prediction[:, 1:] <= 1))


def test_public_predictions_reject_raw_overflow_before_bounded_transformation():
    contract = _contract()
    output = torch.full(
        (1, contract.target_width + len(contract.resource_declarations)),
        torch.finfo(torch.float32).max,
    )
    output = output * 2

    with pytest.raises(ValueError, match="non-finite raw values"):
        public_predictions(output, contract)
