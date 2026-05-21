from dataclasses import asdict, is_dataclass
import os

import torch

from config import PROJECT_ROOT
from version import __version__


MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_FORMAT = "transformer-checkpoint-v1"


def model_path(model_name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, model_name)


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
    }
    if extra:
        payload["extra"] = dict(extra)

    torch.save(payload, full_path)


def load_checkpoint(model_name: str, device):
    payload = torch.load(model_path(model_name), map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
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
