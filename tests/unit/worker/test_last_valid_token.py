import torch
from torch import nn

from app.worker.model.transformer import TransformerModel, public_predictions
from tests.support.consumer_neutral import model_contract


class PassThroughEncoder(nn.Module):
    def forward(self, values, src_key_padding_mask):
        return values


class PassThroughHead(nn.Module):
    def forward_with_shared_representation(self, values):
        return values, values


def test_transformer_selects_last_valid_position_and_all_missing_placeholder():
    contract = model_contract(
        "target-reorder-a-b",
        seq_len=4,
        feature_dim=2,
        hidden=2,
        layers=1,
        dropout=0.0,
        nhead=2,
        mode="strict",
    )
    model = TransformerModel(
        input_dim=2,
        seq_len=4,
        hidden_dim=2,
        layers=1,
        dropout=0.0,
        model_contract=contract,
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
    contract = model_contract(
        "multi-target-shared-resource",
        seq_len=4,
        feature_dim=2,
        hidden=16,
        layers=1,
        dropout=0.0,
        nhead=4,
        mode="relaxed",
    )
    model = TransformerModel(
        input_dim=2,
        seq_len=4,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
        context_mode="relaxed",
    )
    features = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    features[0, 0, :] = float("nan")
    features[1, 1, :] = float("nan")
    features[2, :, :] = float("nan")

    output = model(features)

    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()
    predictions = public_predictions(
        output,
        contract,
    )
    assert predictions.shape == (3, 3)
    assert torch.isfinite(predictions).all()
