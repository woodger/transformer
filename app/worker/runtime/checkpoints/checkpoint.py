import os
from dataclasses import asdict, is_dataclass

import torch

from app.config import PROJECT_ROOT
from app.contracts.worker.v2.config import CHECKPOINT_FORMAT
from app.worker.runtime.checkpoints.atomic import (
    atomic_output_path,
    resolve_artifact_path,
)
from app.worker.runtime.version import __version__

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUPPORTED_CHECKPOINT_FORMATS = {
    "transformer-checkpoint-v1",
    CHECKPOINT_FORMAT,
}


def model_path(model_name: str) -> str:
    return resolve_artifact_path(model_name, MODELS_DIR, label="model path")


def save_checkpoint(
    model_name: str,
    model,
    model_config=None,
    train_config=None,
    extra: dict | None = None,
):
    full_path = model_path(model_name)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": __version__,
        "state_dict": model.state_dict(),
        "model_config": _to_dict(model_config),
        "train_config": _to_dict(train_config),
        "data_schema": _data_schema(model_config),
    }
    if extra:
        payload["extra"] = dict(extra)

    with atomic_output_path(full_path) as temporary_path:
        torch.save(payload, temporary_path)


def load_checkpoint(model_name: str, device):
    payload = torch.load(model_path(model_name), map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        checkpoint_format = payload.get("format")
        if (
            checkpoint_format is not None
            and checkpoint_format not in SUPPORTED_CHECKPOINT_FORMATS
        ):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
        return payload

    return {
        "format": "legacy-state-dict",
        "version": None,
        "state_dict": payload,
        "model_config": None,
        "train_config": None,
    }


def load_checkpoint_metadata(model_name: str, device="cpu") -> dict:
    payload = load_checkpoint(model_name, device)
    return {
        "format": payload.get("format"),
        "version": payload.get("version"),
        "model_config": payload.get("model_config"),
        "train_config": payload.get("train_config"),
        "data_schema": payload.get("data_schema"),
        "extra": payload.get("extra"),
    }


def _data_schema(model_config) -> dict:
    config = _to_dict(model_config) or {}
    seq_len = config.get("seq_len")
    feature_dim = config.get("feature_dim")
    context_mode = config.get("context_mode")
    source_width = (
        seq_len * feature_dim
        if seq_len is not None and feature_dim is not None
        else None
    )
    model_input_feature_dim = (
        feature_dim * 2
        if feature_dim is not None and context_mode == "relaxed"
        else feature_dim
    )

    return {
        "schema_version": 1,
        "tensor_dtype": "float32",
        "src": {
            "column": "src",
            "accepted_element_types": ["float32", "float64"],
            "width": source_width,
        },
        "tgt": {
            "column": "tgt",
            "accepted_element_types": ["float32", "float64"],
            "width": 6,
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
