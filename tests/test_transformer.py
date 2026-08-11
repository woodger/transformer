import pytest
import torch

from app.model.context import (
    context_input_dim,
    context_key_padding_mask,
    context_missingness_ratios,
    context_token_ratios,
    prepare_context_input,
)
from app.model.transformer import TransformerModel


@pytest.fixture(autouse=True)
def torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        yield


def test_transformer_forward_shape():
    batch = 4
    seq_len = 10
    feat_dim = 8
    out_dim = 6

    model = TransformerModel(
        input_dim=feat_dim,
        seq_len=seq_len,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
        out_dim=out_dim,
        nhead=4,
    )

    x = torch.randn(batch, seq_len, feat_dim)
    y = model(x)

    assert y.shape == (batch, out_dim)


def test_transformer_input_dim_matches_context_mode():
    relaxed = TransformerModel(
        input_dim=8,
        seq_len=10,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="relaxed",
    )
    strict = TransformerModel(
        input_dim=8,
        seq_len=10,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="strict",
    )

    assert context_input_dim(8, "relaxed") == 16
    assert relaxed.input_proj.in_features == 16
    assert strict.input_proj.in_features == 8


def test_context_mask_depends_on_context_mode():
    x = torch.tensor([
        [
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    assert context_key_padding_mask(x, "strict").tolist() == [[True, True]]
    assert context_key_padding_mask(x, "relaxed").tolist() == [[False, True]]


def test_context_token_ratios_split_complete_partial_and_empty_tokens():
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
            [4.0, 5.0, 6.0],
        ],
    ])

    ratios = context_token_ratios(x)

    assert ratios["masked_token_ratio"] == 0.25
    assert ratios["complete_token_ratio"] == 0.5
    assert ratios["partial_token_ratio"] == 0.25
    assert ratios["empty_token_ratio"] == 0.25


def test_context_token_ratios_use_selected_masking_mode():
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    assert context_token_ratios(x, "strict")["masked_token_ratio"] == pytest.approx(2 / 3)
    assert context_token_ratios(x, "relaxed")["masked_token_ratio"] == pytest.approx(1 / 3)


def test_context_missingness_ratios_include_values_and_tokens():
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    ratios = context_missingness_ratios(x, "relaxed")

    assert ratios == pytest.approx({
        "nan_ratio": 4 / 9,
        "masked_token_ratio": 1 / 3,
        "complete_token_ratio": 1 / 3,
        "partial_token_ratio": 1 / 3,
        "empty_token_ratio": 1 / 3,
    })


def test_relaxed_keeps_missing_flags_after_nan_to_num():
    x = torch.tensor([
        [
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    values, mask = prepare_context_input(x, "relaxed")

    assert mask.tolist() == [[False, True]]
    assert values.tolist() == [
        [
            [1.0, 0.0, 3.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
    ]


def test_transformer_forward_with_partial_and_full_nan_tokens_is_finite():
    model = TransformerModel(
        input_dim=3,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="relaxed",
    )

    x = torch.randn(2, 3, 3)
    x[0, 1, 2] = float("nan")
    x[0, 2, :] = float("nan")
    x[1, :, :] = float("nan")

    y = model(x)

    assert y.shape == (2, 6)
    assert torch.isfinite(y).all()
