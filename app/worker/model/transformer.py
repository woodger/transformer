from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from app.contracts.semantic.v5 import ModelContract
from app.contracts.semantic.v5.constants import ENCODER_NORMALIZATION_ORDERS
from app.contracts.worker.v20.config import DEFAULT_CONTEXT_MODE
from app.worker.model.context import (
    context_input_dim,
    prepare_context_input,
    validate_context_mode,
)
from app.worker.model.output_head import OutputHead
from app.worker.model.positional_encoding import PositionalEncoding

_ENCODER_BLOCK_BOUNDARIES = (
    "input",
    "attentionResidual",
    "norm1",
    "feedForwardResidual",
    "norm2",
)
_OUTPUT_HEAD_SHARED_BOUNDARIES = ("linear", "gelu", "layerNorm")


def _last_unmasked_indices(key_padding_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(
        key_padding_mask.size(1),
        device=key_padding_mask.device,
    ).expand_as(key_padding_mask)
    return positions.masked_fill(key_padding_mask, -1).max(dim=1).values.clamp(min=0)


class TransformerModel(nn.Module):
    """Отобразить последовательный вход в выбранные публичные и требуемые закрытые head."""

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
        normalization_order: str,
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
        if normalization_order not in ENCODER_NORMALIZATION_ORDERS:
            raise ValueError(
                "normalization_order must be one of: postNorm, preNorm"
            )

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model_contract = model_contract
        self.context_mode = validate_context_mode(context_mode)
        self.normalization_order = normalization_order
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
            norm_first=normalization_order == "preNorm",
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

    def forward_with_representation_flow(
        self,
        features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        """Вернуть вывод и построчные состояния на границах encoder/head."""
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
        encoder_input = self.input_proj(prepared_context.features)
        encoder_input = self.pos(encoder_input)
        captured_layer_outputs: list[torch.Tensor] = []

        def capture_layer_output(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise ValueError("encoder layer output must be a tensor")
            captured_layer_outputs.append(output)

        # Перехватчики сохраняют обычный вызов self.encoder, а не повторяют его по слоям.
        handles = [
            layer.register_forward_hook(capture_layer_output)
            for layer in self.encoder.layers
        ]
        try:
            encoded = self.encoder(
                encoder_input,
                src_key_padding_mask=prepared_context.key_padding_mask,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured_layer_outputs) != len(self.encoder.layers):
            raise ValueError("encoder layers were not observed")
        last_unmasked_indices = _last_unmasked_indices(
            prepared_context.key_padding_mask
        )
        batch_indices = torch.arange(
            encoder_input.size(0),
            device=encoder_input.device,
        )
        encoder_input_rows = encoder_input[
            batch_indices,
            last_unmasked_indices,
        ]
        encoder_layer_rows = tuple(
            values[batch_indices, last_unmasked_indices]
            for values in captured_layer_outputs
        )
        last_valid = encoded[batch_indices, last_unmasked_indices]
        model_output, shared = self.head.forward_with_shared_representation(
            last_valid
        )
        return (
            model_output,
            shared,
            encoder_input_rows,
            encoder_layer_rows,
        )

    def forward_with_encoder_block_flow(
        self,
        features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[tuple[torch.Tensor, ...], ...],
        tuple[torch.Tensor, ...],
    ]:
        """Вернуть состояния encoder и OutputHead.shared."""
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
        encoder_input = self.input_proj(prepared_context.features)
        encoder_input = self.pos(encoder_input)
        layers = tuple(self.encoder.layers)
        norm_first = self.normalization_order == "preNorm"
        if not layers or any(
            not isinstance(layer, torch.nn.TransformerEncoderLayer)
            or layer.norm_first != norm_first
            for layer in layers
        ):
            raise ValueError("encoder block flow requires configured encoder layers")
        shared_modules = tuple(self.head.shared.children())
        if (
            len(shared_modules) != len(_OUTPUT_HEAD_SHARED_BOUNDARIES)
            or not isinstance(shared_modules[0], nn.Linear)
            or not isinstance(shared_modules[1], nn.GELU)
            or not isinstance(shared_modules[2], nn.LayerNorm)
        ):
            raise ValueError("encoder block flow requires the current OutputHead.shared")

        layer_states: list[dict[str, torch.Tensor]] = [
            {} for _layer in layers
        ]
        shared_states: dict[str, torch.Tensor] = {}

        def capture_input(
            layer_index: int,
            boundary: str,
        ) -> Callable[[torch.nn.Module, tuple[object, ...]], None]:
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[object, ...],
            ) -> None:
                layer_states[layer_index][boundary] = _hook_input(
                    inputs,
                    boundary,
                )

            return hook

        def capture_output(
            layer_index: int,
            boundary: str,
        ) -> Callable[[torch.nn.Module, tuple[object, ...], object], None]:
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[object, ...],
                output: object,
            ) -> None:
                layer_states[layer_index][boundary] = _hook_output(
                    output,
                    boundary,
                )

            return hook

        def capture_shared_output(
            boundary: str,
        ) -> Callable[[torch.nn.Module, tuple[object, ...], object], None]:
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[object, ...],
                output: object,
            ) -> None:
                shared_states[boundary] = _hook_output(output, boundary)

            return hook

        handles: list[RemovableHandle] = []
        for layer_index, layer in enumerate(layers):
            norm1 = cast(nn.Module, layer.norm1)
            norm2 = cast(nn.Module, layer.norm2)
            handles.append(layer.register_forward_pre_hook(
                capture_input(layer_index, "input")
            ))
            if self.normalization_order == "postNorm":
                handles.extend((
                    norm1.register_forward_pre_hook(
                        capture_input(layer_index, "attentionResidual")
                    ),
                    norm1.register_forward_hook(
                        capture_output(layer_index, "norm1")
                    ),
                    norm2.register_forward_pre_hook(
                        capture_input(layer_index, "feedForwardResidual")
                    ),
                    norm2.register_forward_hook(
                        capture_output(layer_index, "norm2")
                    ),
                ))
            else:
                handles.extend((
                    norm1.register_forward_hook(
                        capture_output(layer_index, "norm1")
                    ),
                    norm2.register_forward_pre_hook(
                        capture_input(layer_index, "attentionResidual")
                    ),
                    norm2.register_forward_hook(
                        capture_output(layer_index, "norm2")
                    ),
                    layer.register_forward_hook(
                        capture_output(layer_index, "feedForwardResidual")
                    ),
                ))
        for boundary, module in zip(
            _OUTPUT_HEAD_SHARED_BOUNDARIES,
            shared_modules,
            strict=True,
        ):
            handles.append(module.register_forward_hook(
                capture_shared_output(boundary)
            ))
        try:
            encoded = self.encoder(
                encoder_input,
                src_key_padding_mask=prepared_context.key_padding_mask,
            )
            last_unmasked_indices = _last_unmasked_indices(
                prepared_context.key_padding_mask
            )
            batch_indices = torch.arange(
                encoder_input.size(0),
                device=encoder_input.device,
            )
            encoder_input_rows = encoder_input[
                batch_indices,
                last_unmasked_indices,
            ]
            block_rows = tuple(
                tuple(
                    _block_boundary(state, boundary)[
                        batch_indices,
                        last_unmasked_indices,
                    ]
                    for boundary in _ENCODER_BLOCK_BOUNDARIES
                )
                for state in layer_states
            )
            final_layer_boundary = (
                "norm2"
                if self.normalization_order == "postNorm"
                else "feedForwardResidual"
            )
            final_layer_boundary_index = _ENCODER_BLOCK_BOUNDARIES.index(
                final_layer_boundary
            )
            encoder_layer_rows = tuple(
                values[final_layer_boundary_index] for values in block_rows
            )
            last_valid = encoded[batch_indices, last_unmasked_indices]
            model_output, shared = self.head.forward_with_shared_representation(
                last_valid
            )
            shared_rows = tuple(
                _block_boundary(shared_states, boundary)
                for boundary in _OUTPUT_HEAD_SHARED_BOUNDARIES
            )
        finally:
            for handle in handles:
                handle.remove()

        return (
            model_output,
            shared,
            encoder_input_rows,
            encoder_layer_rows,
            block_rows,
            shared_rows,
        )


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


def _hook_input(
    inputs: tuple[object, ...],
    boundary: str,
) -> torch.Tensor:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        raise ValueError(f"{boundary} hook input must be a tensor")
    return inputs[0]


def _hook_output(output: object, boundary: str) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"{boundary} hook output must be a tensor")
    return output


def _block_boundary(
    values: dict[str, torch.Tensor],
    boundary: str,
) -> torch.Tensor:
    try:
        return values[boundary]
    except KeyError as exc:
        raise ValueError(f"encoder block boundary is unavailable: {boundary}") from exc


__all__ = [
    "TransformerModel",
    "apply_transformation",
    "public_predictions",
]
