import torch
import pytest

from app.context import (
    context_input_dim,
    context_key_padding_mask,
    context_token_ratios,
)
from app.transformer import TransformerModel


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
    model = TransformerModel(
        input_dim=8,
        seq_len=10,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="indicators",
    )

    assert context_input_dim(8, "relaxed") == 8
    assert relaxed.input_proj.in_features == 8
    assert model.input_proj.in_features == 16


def test_context_mask_depends_on_context_mode():
    x = torch.tensor([
        [
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    assert context_key_padding_mask(x, "strict").tolist() == [[True, True]]
    assert context_key_padding_mask(x, "relaxed").tolist() == [[False, True]]
    assert context_key_padding_mask(x, "indicators").tolist() == [[False, True]]


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


def test_transformer_forward_with_partial_and_full_nan_tokens_is_finite():
    model = TransformerModel(
        input_dim=3,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="indicators",
    )

    x = torch.randn(2, 3, 3)
    x[0, 1, 2] = float("nan")
    x[0, 2, :] = float("nan")
    x[1, :, :] = float("nan")

    y = model(x)

    assert y.shape == (2, 6)
    assert torch.isfinite(y).all()
