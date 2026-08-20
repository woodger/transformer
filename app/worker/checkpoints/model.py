from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import cast

import torch

from app.contracts.json_types import JsonObject
from app.contracts.worker.v7.config import ModelConfig, TrainConfig
from app.contracts.worker.v7.objective import (
    CHECKPOINT_FORMAT,
    DATA_CONTRACT_ID,
    DATA_CONTRACT_VERSION,
    TARGET_SCHEMA_ID,
    ml_contract,
    objective_config,
)
from app.project import PROJECT_ROOT
from app.worker.checkpoints.atomic import (
    atomic_output_path,
    resolve_artifact_path,
)
from app.worker.runtime.version import __version__

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CheckpointFormatMismatch(ValueError):
    """Checkpoint is valid enough to identify, but belongs to another contract."""


class CheckpointCorrupt(ValueError):
    """Checkpoint claims the current format but has inconsistent metadata."""


def model_path(model_name: str) -> str:
    return resolve_artifact_path(model_name, MODELS_DIR, label="model path")


def save_checkpoint(
    model_name: str,
    model: torch.nn.Module,
    model_config: object = None,
    train_config: object = None,
    data_contract: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    model_config = _model_config(model_config)
    train_config = _train_config(train_config)
    if model_config.feature_dim is None:
        raise ValueError("checkpoint feature dimension is unavailable")
    full_path = model_path(model_name)
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "version": __version__,
        "state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "data_schema": _data_schema(model_config),
        "data_contract": None if data_contract is None else dict(data_contract),
        "ml_contract": ml_contract(train_config),
        "objective_config": objective_config(train_config),
        "extra": {} if extra is None else dict(extra),
    }

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
        )
    except Exception as exc:
        raise CheckpointCorrupt("Checkpoint could not be deserialized") from exc

    if not isinstance(loaded, dict):
        raise CheckpointFormatMismatch("Unsupported checkpoint format")
    payload = _object_dict(cast(object, loaded), "checkpoint")
    checkpoint_format = payload.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise CheckpointFormatMismatch(
            f"Unsupported checkpoint format: {checkpoint_format or 'unknown'}"
        )
    _validate_current_checkpoint(payload)
    return payload


def load_checkpoint_metadata(
    model_name: str,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = load_checkpoint(model_name, device)
    return {
        "format": payload["format"],
        "version": payload["version"],
        "model_config": payload["model_config"],
        "train_config": payload["train_config"],
        "data_schema": payload["data_schema"],
        "data_contract": payload["data_contract"],
        "ml_contract": payload["ml_contract"],
        "objective_config": payload["objective_config"],
        "extra": payload["extra"],
    }


def save_model(
    model_name: str,
    model: torch.nn.Module,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    data_contract: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    save_checkpoint(
        model_name,
        model,
        model_config=model_config,
        train_config=train_config,
        data_contract=data_contract,
        extra=extra,
    )


def load_model(
    model_name: str,
    model: torch.nn.Module,
    device: str | torch.device,
) -> torch.nn.Module:
    checkpoint = load_checkpoint(model_name, device)
    state_value = checkpoint["state_dict"]
    if not isinstance(state_value, Mapping):
        raise ValueError("checkpoint model state must be an object")
    state = cast(Mapping[object, object], state_value)
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("checkpoint model state must contain tensors")
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state))
    return model


def _validate_current_checkpoint(payload: dict[str, object]) -> None:
    required = {
        "format",
        "version",
        "state_dict",
        "model_config",
        "train_config",
        "data_schema",
        "data_contract",
        "ml_contract",
        "objective_config",
        "extra",
    }
    if set(payload) != required:
        raise CheckpointCorrupt("Checkpoint has invalid fields")
    if not isinstance(payload["version"], str) or not payload["version"]:
        raise CheckpointCorrupt("Checkpoint version is invalid")
    if not isinstance(payload["state_dict"], dict):
        raise CheckpointCorrupt("Checkpoint model state is invalid")
    if not isinstance(payload["extra"], dict):
        raise CheckpointCorrupt("Checkpoint extra metadata is invalid")
    try:
        model_config = _model_config(payload["model_config"])
        train_config = _train_config(payload["train_config"])
    except (TypeError, ValueError) as exc:
        raise CheckpointCorrupt("Checkpoint configuration is invalid") from exc
    if model_config.feature_dim is None:
        raise CheckpointCorrupt("Checkpoint feature dimension is unavailable")
    if payload["data_schema"] != _data_schema(model_config):
        raise CheckpointCorrupt("Checkpoint data schema is inconsistent")
    if payload["ml_contract"] != ml_contract(train_config):
        raise CheckpointCorrupt("Checkpoint ML contract is inconsistent")
    if payload["objective_config"] != objective_config(train_config):
        raise CheckpointCorrupt("Checkpoint objective configuration is inconsistent")
    data_contract_value = payload["data_contract"]
    data_contract = (
        None
        if data_contract_value is None
        else _object_dict(data_contract_value, "checkpoint data contract")
    )
    if data_contract is not None:
        profile = data_contract.get("profile")
        digest = data_contract.get("dataContractSha256")
        if (
            set(data_contract) != {
                "id",
                "version",
                "profile",
                "dataContractSha256",
                "seqLen",
                "featureDim",
                "targetSchemaId",
            }
            or data_contract.get("id") != DATA_CONTRACT_ID
            or data_contract.get("version") != DATA_CONTRACT_VERSION
            or not isinstance(profile, str)
            or _PROFILE.fullmatch(profile) is None
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or data_contract.get("targetSchemaId") != TARGET_SCHEMA_ID
            or data_contract.get("seqLen") != model_config.seq_len
            or data_contract.get("featureDim") != model_config.feature_dim
        ):
            raise CheckpointCorrupt("Checkpoint data contract is inconsistent")


def _data_schema(model_config: ModelConfig) -> JsonObject:
    seq_len = model_config.seq_len
    feature_dim = model_config.feature_dim
    context_mode = model_config.context_mode
    source_width = (
        seq_len * feature_dim if feature_dim is not None else None
    )
    model_input_feature_dim = (
        feature_dim * 2
        if feature_dim is not None and context_mode == "relaxed"
        else feature_dim
    )

    return {
        "schema_version": 2,
        "tensor_dtype": "float32",
        "src": {
            "column": "src",
            "accepted_element_types": ["float32"],
            "width": source_width,
        },
        "tgt": {
            "column": "tgt",
            "accepted_element_types": ["float32"],
            "width": 6,
            "target_schema_id": TARGET_SCHEMA_ID,
        },
        "feature_dim": feature_dim,
        "model_input_feature_dim": model_input_feature_dim,
        "context_mode": context_mode,
        "normalization": None,
        "missing": {
            "nan_fill": 0.0,
            "flags": "per-feature" if context_mode == "relaxed" else "none",
        },
    }


def _model_config(value: object) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    config = ModelConfig.from_dict(value)
    if config is None:
        raise ValueError("model configuration is required")
    return config


def _train_config(value: object) -> TrainConfig:
    if isinstance(value, TrainConfig):
        return value
    config = TrainConfig.from_dict(value)
    if config is None:
        raise ValueError("training configuration is required")
    return config


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
    "load_checkpoint_metadata",
    "model_path",
    "save_checkpoint",
]
