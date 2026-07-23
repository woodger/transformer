import torch
from torch import nn

from app.model.transformer import TransformerModel


class PassThroughEncoder(nn.Module):
    def forward(self, values, src_key_padding_mask):
        return values


def test_transformer_selects_last_valid_position_and_all_missing_placeholder():
    model = TransformerModel(
        input_dim=2,
        seq_len=4,
        hidden_dim=2,
        layers=1,
        dropout=0.0,
        out_dim=2,
        nhead=2,
        context_mode="strict",
    )
    model.input_proj = nn.Identity()
    model.pos = nn.Identity()
    model.encoder = PassThroughEncoder()
    model.head = nn.Identity()
    missing = [float("nan"), float("nan")]
    x = torch.tensor([
        [missing, [10.0, 11.0], [20.0, 21.0], missing],
        [[30.0, 31.0], missing, [40.0, 41.0], missing],
        [missing, [50.0, 51.0], missing, missing],
        [missing, missing, missing, missing],
    ])

    output = model(x)

    assert output.tolist() == [
        [20.0, 21.0],
        [40.0, 41.0],
        [50.0, 51.0],
        [0.0, 0.0],
    ]


def test_transformer_forward_with_missing_tokens_is_finite():
    model = TransformerModel(
        input_dim=2,
        seq_len=4,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="relaxed",
    )
    x = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    x[0, 0, :] = float("nan")
    x[1, 1, :] = float("nan")
    x[2, :, :] = float("nan")

    output = model(x)

    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()
