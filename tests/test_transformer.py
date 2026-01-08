import torch

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
