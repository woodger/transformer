from __future__ import annotations

import os
from collections.abc import Mapping
from typing import cast

import torch

from app.contracts.checkpoint.v10 import (
    CHECKPOINT_FORMAT,
    validate_checkpoint_document,
)
from app.contracts.json_types import JsonObject
from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v17.config import ModelConfig
from app.contracts.worker.v17.model_definition import resolved_semantic_digests
from app.project import PROJECT_ROOT
from app.worker.checkpoints.atomic import atomic_output_path, resolve_artifact_path
from app.worker.checkpoints.checkpoint_corrupt import CheckpointCorrupt
from app.worker.checkpoints.checkpoint_format_mismatch import (
    CheckpointFormatMismatch,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def model_path(model_name: str) -> str:
    return resolve_artifact_path(model_name, MODELS_DIR, label="model path")


def save_checkpoint(
    model_name: str,
    model: torch.nn.Module,
    *,
    metadata: JsonObject,
) -> None:
    validate_checkpoint_document(metadata, "checkpoint-metadata")
    payload: dict[str, object] = {
        "metadata": dict(metadata),
        "state_dict": model.state_dict(),
    }
    full_path = model_path(model_name)
    with atomic_output_path(full_path) as temporary_path:
        torch.save(payload, temporary_path)


def load_checkpoint(
    model_name: str,
    device: str | torch.device,
) -> dict[str, object]:
    try:
        loaded: object = torch.load(
            model_path(model_name),
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise CheckpointCorrupt("checkpoint could not be deserialized") from exc
    if not isinstance(loaded, Mapping):
        raise CheckpointFormatMismatch("unsupported checkpoint format")
    payload = _object_dict(
        cast(Mapping[object, object], loaded),
        "checkpoint",
    )
    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise CheckpointFormatMismatch("unsupported checkpoint format")
    metadata = _object_dict(
        cast(Mapping[object, object], metadata_value),
        "checkpoint metadata",
    )
    checkpoint_format = metadata.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise CheckpointFormatMismatch(
            f"unsupported checkpoint format: {checkpoint_format or 'unknown'}"
        )
    if set(payload) != {"metadata", "state_dict"}:
        raise CheckpointCorrupt("checkpoint has invalid fields")
    try:
        validate_checkpoint_document(metadata, "checkpoint-metadata")
        _tensor_state_dict(payload["state_dict"])
        model_contract = ModelContract.from_document(metadata["modelContract"])
        data_contract = _object_dict(
            cast(Mapping[object, object], metadata["dataContract"]),
            "checkpoint data contract",
        )
        data_digest = data_contract.get("dataContractSha256")
        if not isinstance(data_digest, str):
            raise ValueError("checkpoint data contract digest is invalid")
        model_config = ModelConfig.from_manifest(metadata.get("modelConfig"))
        if (
            data_contract.get("seqLen") != model_config.seq_len
            or data_contract.get("featureDim") != model_config.feature_dim
            or model_config.to_tuning() != model_contract.model_tuning
        ):
            raise ValueError("checkpoint data and model geometry differ")
        if metadata.get("predictionDefinition") != model_contract.prediction_definition(
            model_config.seq_len
        ):
            raise ValueError("checkpoint prediction definition is inconsistent")
        semantic_digests = _object_dict(
            cast(Mapping[object, object], metadata["semanticDigests"]),
            "checkpoint semantic digests",
        )
        if semantic_digests != resolved_semantic_digests(
            model_contract,
            data_digest,
            model_config,
        ):
            raise ValueError("checkpoint semantic digests are inconsistent")
    except (TypeError, ValueError) as exc:
        raise CheckpointCorrupt("checkpoint contents are invalid") from exc
    payload["metadata"] = metadata
    return payload


def save_model(
    model_name: str,
    model: torch.nn.Module,
    *,
    metadata: JsonObject,
) -> None:
    save_checkpoint(model_name, model, metadata=metadata)


def _tensor_state_dict(value: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint model state must be an object")
    state = cast(Mapping[object, object], value)
    if not all(
        isinstance(key, str) and isinstance(item, torch.Tensor)
        for key, item in state.items()
    ):
        raise ValueError("checkpoint model state must contain tensors")
    return cast(Mapping[str, torch.Tensor], state)


def _object_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointCorrupt",
    "CheckpointFormatMismatch",
    "load_checkpoint",
    "model_path",
    "save_checkpoint",
    "save_model",
]
