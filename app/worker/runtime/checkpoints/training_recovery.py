from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, is_dataclass

import torch

from app.contracts.worker.v3.config import TrainConfig
from app.contracts.worker.v3.objective import (
    TRAINING_RECOVERY_FORMAT,
    ml_contract,
    objective_config,
    objective_config_sha256,
)
from app.worker.runtime.version import __version__

_HASH_CHUNK_BYTES = 1024 * 1024


def save_training_recovery(
    path: str,
    trainer,
    *,
    generation: int,
    config_hash: str,
    manifest_hash: str,
) -> dict:
    """Atomically persist one complete global-epoch recovery point."""

    _positive(generation, "generation")
    _digest(config_hash, "config_hash")
    _digest(manifest_hash, "manifest_hash")
    state = trainer.recovery_state_dict()
    progress = state["training_state"]
    payload = {
        "format": TRAINING_RECOVERY_FORMAT,
        "service_version": __version__,
        "generation": generation,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "completed_epochs": progress["global_epoch"],
        "global_step": progress["train_step"],
        "training_complete": state["training_complete"],
        "model_config": _to_dict(trainer.model_config),
        "train_config": _to_dict(trainer.train_config),
        "data_contract": (
            None
            if trainer.data_contract is None
            else dict(trainer.data_contract)
        ),
        "ml_contract": ml_contract(trainer.train_config),
        "objective_config": objective_config(trainer.train_config),
        "objective_config_sha256": objective_config_sha256(
            trainer.train_config
        ),
        "trainer_state": state,
    }
    target = os.path.abspath(os.fspath(path))
    _atomic_torch_save(target, payload)
    byte_count = os.path.getsize(target)
    return {
        "format": TRAINING_RECOVERY_FORMAT,
        "generation": generation,
        "completed_epochs": progress["global_epoch"],
        "global_step": progress["train_step"],
        "training_complete": state["training_complete"],
        "bytes": byte_count,
        "sha256": sha256_file(target),
    }


def load_training_recovery(
    path: str,
    device,
    *,
    expected_config_hash: str,
    expected_manifest_hash: str,
    expected_objective_config_sha256: str,
    expected_data_contract_sha256: str | None = None,
) -> dict:
    """Load and validate one server-owned training recovery checkpoint."""

    _digest(expected_config_hash, "expected_config_hash")
    _digest(expected_manifest_hash, "expected_manifest_hash")
    _digest(
        expected_objective_config_sha256,
        "expected_objective_config_sha256",
    )
    if expected_data_contract_sha256 is not None:
        _digest(
            expected_data_contract_sha256,
            "expected_data_contract_sha256",
        )
    payload = torch.load(
        os.path.abspath(os.fspath(path)),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("training recovery checkpoint must be an object")
    required = {
        "format",
        "service_version",
        "generation",
        "config_hash",
        "manifest_hash",
        "completed_epochs",
        "global_step",
        "training_complete",
        "model_config",
        "train_config",
        "data_contract",
        "ml_contract",
        "objective_config",
        "objective_config_sha256",
        "trainer_state",
    }
    if set(payload) != required:
        raise ValueError("training recovery checkpoint has invalid fields")
    if payload["format"] != TRAINING_RECOVERY_FORMAT:
        raise ValueError(
            f"Unsupported training recovery format: {payload['format']}"
        )
    if payload["config_hash"] != expected_config_hash:
        raise ValueError(
            "training recovery configuration does not match the job"
        )
    if payload["manifest_hash"] != expected_manifest_hash:
        raise ValueError(
            "training recovery inputs do not match the closed job"
        )
    try:
        train_config = TrainConfig.from_dict(payload["train_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training recovery objective configuration is invalid"
        ) from exc
    if train_config is None or (
        payload["objective_config_sha256"]
        != expected_objective_config_sha256
        or payload["objective_config_sha256"]
        != objective_config_sha256(train_config)
        or payload["objective_config"] != objective_config(train_config)
        or payload["ml_contract"] != ml_contract(train_config)
    ):
        raise ValueError(
            "training recovery objective configuration does not match the job"
        )
    if expected_data_contract_sha256 is not None and (
        not isinstance(payload["data_contract"], dict)
        or payload["data_contract"].get("dataContractSha256")
        != expected_data_contract_sha256
    ):
        raise ValueError(
            "training recovery data contract does not match the job"
        )
    _positive(payload["generation"], "generation")
    _positive(payload["completed_epochs"], "completed_epochs")
    _nonnegative(payload["global_step"], "global_step")
    if payload["generation"] != payload["completed_epochs"]:
        raise ValueError(
            "training recovery generation must equal completed epochs"
        )
    if not isinstance(payload["training_complete"], bool):
        raise ValueError(
            "training recovery completion marker must be a boolean"
        )
    state = payload["trainer_state"]
    if not isinstance(state, dict):
        raise ValueError("training recovery trainer state is invalid")
    progress = state.get("training_state")
    if not isinstance(progress, dict) or (
        progress.get("global_epoch") != payload["completed_epochs"]
        or progress.get("train_step") != payload["global_step"]
    ):
        raise ValueError(
            "training recovery progress metadata is inconsistent"
        )
    if state.get("training_complete") is not payload["training_complete"]:
        raise ValueError(
            "training recovery completion metadata is inconsistent"
        )
    return payload


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(
            lambda: source.read(_HASH_CHUNK_BYTES),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: str, payload: dict) -> None:
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


def _digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


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
    "TRAINING_RECOVERY_FORMAT",
    "load_training_recovery",
    "save_training_recovery",
    "sha256_file",
]
