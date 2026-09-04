from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.contracts.ml import canonical_targets
from app.contracts.worker.v11.config import DEFAULT_CONTEXT_MODE
from app.worker.model.context import (
    context_input_dim,
    prepare_context_input,
    validate_context_mode,
)
from app.worker.model.positional_encoding import PositionalEncoding


def _last_unmasked_indices(key_padding_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(
        key_padding_mask.size(1),
        device=key_padding_mask.device,
    ).expand_as(key_padding_mask)
    return positions.masked_fill(key_padding_mask, -1).max(dim=1).values.clamp(min=0)


class TradingHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        targets: Sequence[str],
        *,
        include_return_scale: bool,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")

        self.hidden_dim = hidden_dim
        self.targets = canonical_targets(targets)
        self.include_return_scale = include_return_scale
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.public_heads = nn.ModuleDict({
            target: nn.Linear(128, 1)
            for target in self.targets
        })
        self.return_scale_head = (
            nn.Linear(128, 1)
            if include_return_scale
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _shared = self.forward_with_shared_representation(hidden_states)
        return output

    def forward_with_shared_representation(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_dim:
            raise ValueError("hidden states must have shape [batch, hidden]")

        shared = self.shared(hidden_states)
        public = [
            _internal_public_value(target, self.public_heads[target](shared))
            for target in self.targets
        ]
        values = public
        if self.return_scale_head is not None:
            values = [
                *values,
                F.softplus(self.return_scale_head(shared)) + 1e-6,
            ]
        return torch.cat(values, dim=1), shared


class TransformerModel(nn.Module):
    """Map sequence inputs to selected public heads and required private heads."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        targets: Sequence[str],
        *,
        include_return_scale: bool,
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
        self.targets = canonical_targets(targets)
        self.include_return_scale = include_return_scale
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
        self.head = TradingHead(
            hidden_dim,
            self.targets,
            include_return_scale=include_return_scale,
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
    targets: Sequence[str],
    *,
    include_return_scale: bool = False,
) -> torch.Tensor:
    selected = canonical_targets(targets)
    expected_width = len(selected) + int(include_return_scale)
    if model_output.ndim != 2 or model_output.shape[1] != expected_width:
        raise ValueError(
            f"model output must have shape [rows, {expected_width}]"
        )
    return torch.stack(
        tuple(
            _public_value(target, model_output[:, index])
            for index, target in enumerate(selected)
        ),
        dim=1,
    )


def _internal_public_value(target: str, value: torch.Tensor) -> torch.Tensor:
    if target == "MeanReturn":
        return torch.tanh(value)
    if target in {"SigmaReturn", "VolatilityNext"}:
        return torch.sigmoid(value)
    return value


def _public_value(target: str, value: torch.Tensor) -> torch.Tensor:
    if target in {"ProbTP", "ProbSL", "HittingProbTP"}:
        return torch.sigmoid(value)
    return value


__all__ = ["TradingHead", "TransformerModel", "public_predictions"]
