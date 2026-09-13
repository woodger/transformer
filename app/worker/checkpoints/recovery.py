from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from typing import Protocol, cast

import torch

from app.contracts.checkpoint.v7 import (
    RECOVERY_FORMAT,
    validate_checkpoint_document,
)
from app.contracts.json_types import JsonObject

_HASH_CHUNK_BYTES = 1024 * 1024


class RecoveryTrainer(Protocol):
    def recovery_state_dict(self) -> dict[str, object]: ...


def save_training_recovery(
    path: str,
    trainer: RecoveryTrainer,
    *,
    metadata: JsonObject,
) -> JsonObject:
    """Atomically persist one complete global-epoch recovery point."""

    validate_checkpoint_document(metadata, "checkpoint-metadata")
    payload: dict[str, object] = {
        "metadata": dict(metadata),
        "trainer_state": trainer.recovery_state_dict(),
    }
    target = os.path.abspath(os.fspath(path))
    _atomic_torch_save(target, payload)
    byte_count = os.path.getsize(target)
    progress = _object_dict(metadata["progress"], "checkpoint progress")
    return {
        "format": RECOVERY_FORMAT,
        "generation": _positive(metadata["generation"], "generation"),
        "completedEpochs": _positive(
            progress["completedEpochs"],
            "completed epochs",
        ),
        "globalStep": _nonnegative(progress["globalStep"], "global step"),
        "trainingComplete": _boolean(
            progress["trainingComplete"],
            "training complete",
        ),
        "byteCount": byte_count,
        "checkpointSha256": sha256_file(target),
    }


def load_training_recovery(
    path: str,
    device: str | torch.device,
    *,
    descriptor: JsonObject,
) -> dict[str, object]:
    """Load a recovery payload and enforce every descriptor fence."""

    loaded: object = torch.load(
        os.path.abspath(os.fspath(path)),
        map_location=device,
        weights_only=False,
    )
    payload = _object_dict(loaded, "training recovery checkpoint")
    if set(payload) != {"metadata", "trainer_state"}:
        raise ValueError("training recovery checkpoint has invalid fields")
    metadata = _object_dict(payload["metadata"], "checkpoint metadata")
    validate_checkpoint_document(metadata, "checkpoint-metadata")
    progress = _object_dict(metadata["progress"], "checkpoint progress")
    expected = {
        "jobId": descriptor["jobId"],
        "generation": descriptor["generation"],
        "jobConfigSha256": descriptor["jobConfigSha256"],
        "semanticDigests": descriptor["semanticDigests"],
        "manifestSha256": descriptor["manifestSha256"],
        "progress": descriptor["progress"],
    }
    actual = {
        "jobId": metadata["jobId"],
        "generation": metadata["generation"],
        "jobConfigSha256": metadata["jobConfigSha256"],
        "semanticDigests": metadata["semanticDigests"],
        "manifestSha256": metadata["manifestSha256"],
        "progress": progress,
    }
    if actual != expected:
        raise ValueError("training recovery checkpoint fence does not match")
    trainer_state = _object_dict(
        payload["trainer_state"],
        "training recovery trainer state",
    )
    training_state = _object_dict(
        trainer_state.get("training_state"),
        "training recovery progress",
    )
    if (
        _positive(metadata["generation"], "generation")
        != _positive(progress["completedEpochs"], "completed epochs")
        or _nonnegative(
            training_state.get("global_epoch"),
            "training global epoch",
        )
        != _positive(progress["completedEpochs"], "completed epochs")
        or _nonnegative(
            training_state.get("train_step"),
            "training step",
        )
        != _nonnegative(progress["globalStep"], "global step")
        or _boolean(
            trainer_state.get("training_complete"),
            "training completion marker",
        )
        != _boolean(progress["trainingComplete"], "training complete")
    ):
        raise ValueError(
            "training recovery trainer progress does not match checkpoint metadata"
        )
    payload["metadata"] = metadata
    return payload


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: str, payload: dict[str, object]) -> None:
    parent = os.path.dirname(path) or os.curdir
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: str) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _object_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


__all__ = [
    "RECOVERY_FORMAT",
    "load_training_recovery",
    "save_training_recovery",
    "sha256_file",
]
