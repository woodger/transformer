import torch
from torch import nn

from app.contracts.ml import TARGET_IDENTITIES
from app.worker.model.transformer import TransformerModel, public_predictions


class PassThroughEncoder(nn.Module):
    def forward(self, values, src_key_padding_mask):
        return values


class PassThroughHead(nn.Module):
    def forward_with_shared_representation(self, values):
        return values, values


def test_transformer_selects_last_valid_position_and_all_missing_placeholder():
    model = TransformerModel(
        input_dim=2,
        seq_len=4,
        hidden_dim=2,
        layers=1,
        dropout=0.0,
        targets=TARGET_IDENTITIES[:2],
        include_return_scale=False,
        nhead=2,
        context_mode="strict",
    )
    model.input_proj = nn.Identity()
    model.pos = nn.Identity()
    model.encoder = PassThroughEncoder()
    model.head = PassThroughHead()
    missing = [float("nan"), float("nan")]
    features = torch.tensor([
        [missing, [10.0, 11.0], [20.0, 21.0], missing],
        [[30.0, 31.0], missing, [40.0, 41.0], missing],
        [missing, [50.0, 51.0], missing, missing],
        [missing, missing, missing, missing],
    ])

    output = model(features)

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
        targets=TARGET_IDENTITIES,
        include_return_scale=True,
        nhead=4,
        context_mode="relaxed",
    )
    features = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    features[0, 0, :] = float("nan")
    features[1, 1, :] = float("nan")
    features[2, :, :] = float("nan")

    output = model(features)

    assert output.shape == (3, 7)
    assert torch.isfinite(output).all()
    predictions = public_predictions(
        output,
        TARGET_IDENTITIES,
        include_return_scale=True,
    )
    assert predictions.shape == (3, 6)
    assert torch.isfinite(predictions).all()
