from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass

import torch

from app.config import PROJECT_ROOT
from app.contracts.worker.v3.config import ModelConfig, TrainConfig
from app.contracts.worker.v3.objective import (
    CHECKPOINT_FORMAT,
    TARGET_SCHEMA_ID,
    ml_contract,
    objective_config,
)
from app.worker.runtime.checkpoints.atomic import (
    atomic_output_path,
    resolve_artifact_path,
)
from app.worker.runtime.version import __version__

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class CheckpointFormatMismatch(ValueError):
    """Checkpoint is valid enough to identify, but belongs to another contract."""


class CheckpointCorrupt(ValueError):
    """Checkpoint claims the current format but has inconsistent metadata."""


def model_path(model_name: str) -> str:
    return resolve_artifact_path(model_name, MODELS_DIR, label="model path")


def save_checkpoint(
    model_name: str,
    model,
    model_config=None,
    train_config=None,
    data_contract: dict | None = None,
    extra: dict | None = None,
):
    model_config = _model_config(model_config)
    train_config = _train_config(train_config)
    if model_config.feature_dim is None:
        raise ValueError("checkpoint feature dimension is unavailable")
    full_path = model_path(model_name)
    payload = {
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


def load_checkpoint(model_name: str, device):
    try:
        payload = torch.load(model_path(model_name), map_location=device)
    except Exception as exc:
        raise CheckpointCorrupt("Checkpoint could not be deserialized") from exc

    if not isinstance(payload, dict):
        raise CheckpointFormatMismatch("Unsupported checkpoint format")
    checkpoint_format = payload.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise CheckpointFormatMismatch(
            f"Unsupported checkpoint format: {checkpoint_format or 'unknown'}"
        )
    _validate_current_checkpoint(payload)
    return payload


def load_checkpoint_metadata(model_name: str, device="cpu") -> dict:
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


def _validate_current_checkpoint(payload: dict) -> None:
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
    data_contract = payload["data_contract"]
    if data_contract is not None and (
        not isinstance(data_contract, dict)
        or data_contract.get("targetSchemaId") != TARGET_SCHEMA_ID
        or data_contract.get("seqLen") != model_config.seq_len
        or data_contract.get("featureDim") != model_config.feature_dim
    ):
        raise CheckpointCorrupt("Checkpoint data contract is inconsistent")


def _data_schema(model_config: ModelConfig) -> dict:
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


def _model_config(value) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    config = ModelConfig.from_dict(_to_dict(value))
    if config is None:
        raise ValueError("model configuration is required")
    return config


def _train_config(value) -> TrainConfig:
    if isinstance(value, TrainConfig):
        return value
    config = TrainConfig.from_dict(_to_dict(value))
    if config is None:
        raise ValueError("training configuration is required")
    return config


def _to_dict(value):
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointCorrupt",
    "CheckpointFormatMismatch",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "model_path",
    "save_checkpoint",
]
