from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v16.config import DEFAULT_CONTEXT_MODE
from app.worker.model.context import (
    context_input_dim,
    prepare_context_input,
    validate_context_mode,
)
from app.worker.model.output_head import OutputHead
from app.worker.model.positional_encoding import PositionalEncoding


def _last_unmasked_indices(key_padding_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(
        key_padding_mask.size(1),
        device=key_padding_mask.device,
    ).expand_as(key_padding_mask)
    return positions.masked_fill(key_padding_mask, -1).max(dim=1).values.clamp(min=0)


class TransformerModel(nn.Module):
    """Map sequence inputs to selected public heads and required private heads."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        model_contract: ModelContract,
        *,
        nhead: int = 8,
        context_mode: str = DEFAULT_CONTEXT_MODE,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if layers <= 0:
            raise ValueError("layers must be a positive integer")
        if nhead <= 0 or hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by a positive nhead")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model_contract = model_contract
        self.context_mode = validate_context_mode(context_mode)
        self.input_proj = nn.Linear(
            context_input_dim(input_dim, self.context_mode),
            hidden_dim,
        )
        self.pos = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.head = OutputHead(
            hidden_dim,
            model_contract.target_width,
            tuple(
                str(resource["resourceClass"])
                for resource in model_contract.resource_declarations
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _shared = self.forward_with_shared_representation(features)
        return output

    def forward_with_shared_representation(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, sequence, features]")
        if features.shape[1] != self.seq_len:
            raise ValueError("sequence length differs from model configuration")
        if features.shape[2] != self.input_dim:
            raise ValueError("feature dimension differs from model configuration")
        if features.dtype != torch.float32:
            raise ValueError("features must use float32")

        prepared_context = prepare_context_input(
            features,
            self.context_mode,
        )
        encoded_features = self.input_proj(prepared_context.features)
        encoded_features = self.pos(encoded_features)
        encoded = self.encoder(
            encoded_features,
            src_key_padding_mask=prepared_context.key_padding_mask,
        )
        last_unmasked_indices = _last_unmasked_indices(
            prepared_context.key_padding_mask
        )
        batch_indices = torch.arange(
            encoded_features.size(0),
            device=encoded_features.device,
        )
        last_valid = encoded[batch_indices, last_unmasked_indices]
        return self.head.forward_with_shared_representation(last_valid)


def public_predictions(
    model_output: torch.Tensor,
    model_contract: ModelContract,
) -> torch.Tensor:
    expected_width = (
        model_contract.target_width + len(model_contract.resource_declarations)
    )
    if model_output.ndim != 2 or model_output.shape[1] != expected_width:
        raise ValueError(
            f"model output must have shape [rows, {expected_width}]"
        )
    if not bool(torch.isfinite(model_output).all()):
        raise ValueError("model output contains non-finite raw values")
    values: list[torch.Tensor] = []
    for index, slot in enumerate(model_contract.target_slots):
        raw_value = model_output[:, index]
        positive_class_weight = model_contract.positive_class_weight_for_target(
            index
        )
        if positive_class_weight is not None:
            values.append(torch.sigmoid(
                raw_value - torch.log(raw_value.new_tensor(positive_class_weight))
            ))
            continue
        values.append(apply_transformation(
            raw_value,
            cast(str, slot["publicPredictionTransformation"]),
        ))
    return torch.stack(values, dim=1)


def apply_transformation(value: torch.Tensor, transformation: str) -> torch.Tensor:
    if transformation == "Identity":
        return value
    if transformation == "Tanh":
        return torch.tanh(value)
    if transformation == "Sigmoid":
        return torch.sigmoid(value)
    raise ValueError(f"unsupported transformation: {transformation}")


__all__ = [
    "TransformerModel",
    "apply_transformation",
    "public_predictions",
]
