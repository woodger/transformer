import torch

from app.context import context_key_padding_mask, context_valid_token_ratio
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


def test_transformer_appends_missing_indicators():
    model = TransformerModel(
        input_dim=8,
        seq_len=10,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )

    assert model.input_proj.in_features == 16


def test_context_mask_keeps_partial_nan_tokens_valid():
    x = torch.tensor([
        [
            [1.0, float("nan"), 3.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ])

    mask = context_key_padding_mask(x)

    assert mask.tolist() == [[False, True]]
    assert context_valid_token_ratio(x) == 0.5


def test_transformer_forward_with_partial_and_full_nan_tokens_is_finite():
    model = TransformerModel(
        input_dim=3,
        seq_len=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )

    x = torch.randn(2, 3, 3)
    x[0, 1, 2] = float("nan")
    x[0, 2, :] = float("nan")
    x[1, :, :] = float("nan")

    y = model(x)

    assert y.shape == (2, 6)
    assert torch.isfinite(y).all()
