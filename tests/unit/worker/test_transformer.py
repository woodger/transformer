import pytest
import torch

from app.worker.model.context import (
    context_input_dim,
    context_key_padding_mask,
    context_missingness_ratios,
    context_token_ratios,
    prepare_context_input,
)
from app.worker.model.transformer import TransformerModel, public_predictions
from tests.support.consumer_neutral import model_contract


def _contract(
    *,
    seq_len: int,
    feature_dim: int,
    mode: str = "relaxed",
):
    return model_contract(
        "multi-target-shared-resource",
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden=32,
        layers=1,
        dropout=0.0,
        nhead=4,
        mode=mode,
    )


@pytest.fixture(autouse=True)
def torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        yield


def test_transformer_forward_shape():
    batch = 4
    seq_len = 10
    feat_dim = 8
    contract = _contract(seq_len=seq_len, feature_dim=feat_dim)

    model = TransformerModel(
        input_dim=feat_dim,
        seq_len=seq_len,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
    )

    features = torch.randn(batch, seq_len, feat_dim)
    model_output = model(features)

    assert model_output.shape == (batch, 4)
    assert public_predictions(
        model_output,
        contract,
    ).shape == (batch, 3)


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
    contract = _contract(seq_len=3, feature_dim=8)
    model = TransformerModel(
        input_dim=8,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
    )

    with pytest.raises(ValueError, match=message):
        model(features)


def test_transformer_rejects_invalid_attention_dimensions():
    contract = _contract(seq_len=3, feature_dim=8)
    with pytest.raises(ValueError, match="divisible"):
        TransformerModel(
            input_dim=8,
            seq_len=3,
            hidden_dim=30,
            layers=1,
            dropout=0.0,
            model_contract=contract,
            nhead=8,
        )


def test_transformer_input_dim_matches_context_mode():
    relaxed_contract = _contract(seq_len=10, feature_dim=8)
    strict_contract = _contract(
        seq_len=10,
        feature_dim=8,
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
    contract = _contract(seq_len=3, feature_dim=3)
    model = TransformerModel(
        input_dim=3,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
        context_mode="relaxed",
    )

    features = torch.randn(2, 3, 3)
    features[0, 1, 2] = float("nan")
    features[0, 2, :] = float("nan")
    features[1, :, :] = float("nan")

    model_output = model(features)

    assert model_output.shape == (2, 4)
    assert torch.isfinite(model_output).all()


def test_public_predictions_apply_declared_slot_transformations():
    contract = _contract(seq_len=3, feature_dim=3)
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
    contract = _contract(seq_len=3, feature_dim=3)
    output = torch.full(
        (1, contract.target_width + len(contract.resource_declarations)),
        torch.finfo(torch.float32).max,
    )
    output = output * 2

    with pytest.raises(ValueError, match="non-finite raw values"):
        public_predictions(output, contract)
