from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from typing import Protocol, cast

import torch

from app.contracts.json_types import JsonObject
from app.contracts.ml import TARGET_SCHEMA_ID
from app.contracts.worker.v9.config import ModelConfig, TrainConfig
from app.contracts.worker.v9.objective import (
    TRAINING_RECOVERY_FORMAT,
    ObjectiveConfig,
    ml_contract,
    objective_config_sha256,
    objective_from_ml_contract,
)
from app.worker.runtime.version import __version__

_HASH_CHUNK_BYTES = 1024 * 1024


class RecoveryTrainer(Protocol):
    @property
    def model_config(self) -> ModelConfig | None: ...

    @property
    def train_config(self) -> TrainConfig | None: ...

    @property
    def data_contract(self) -> Mapping[str, object] | None: ...

    @property
    def objective(self) -> ObjectiveConfig: ...

    def recovery_state_dict(self) -> dict[str, object]: ...


def save_training_recovery(
    path: str,
    trainer: RecoveryTrainer,
    *,
    generation: int,
    config_hash: str,
    manifest_hash: str,
) -> JsonObject:
    """Atomically persist one complete global-epoch recovery point."""

    _positive(generation, "generation")
    _digest(config_hash, "config_hash")
    _digest(manifest_hash, "manifest_hash")
    model_config = trainer.model_config
    train_config = trainer.train_config
    if model_config is None or train_config is None:
        raise ValueError("training recovery configuration is unavailable")
    state = trainer.recovery_state_dict()
    objective = trainer.objective
    target_schema_id = _target_schema_id(trainer.data_contract)
    progress = _object_dict(
        state.get("training_state"),
        "training recovery progress",
    )
    completed_epochs = _positive(
        progress.get("global_epoch"),
        "completed_epochs",
    )
    global_step = _nonnegative(progress.get("train_step"), "global_step")
    training_complete = _boolean(
        state.get("training_complete"),
        "training_complete",
    )
    payload: dict[str, object] = {
        "format": TRAINING_RECOVERY_FORMAT,
        "service_version": __version__,
        "generation": generation,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "training_complete": training_complete,
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "data_contract": (
            None
            if trainer.data_contract is None
            else dict(trainer.data_contract)
        ),
        "ml_contract": ml_contract(
            objective,
            target_schema_id=target_schema_id,
        ),
        "objective": objective.to_document(),
        "objective_config_sha256": objective_config_sha256(
            objective
        ),
        "trainer_state": state,
    }
    target = os.path.abspath(os.fspath(path))
    _atomic_torch_save(target, payload)
    byte_count = os.path.getsize(target)
    return {
        "format": TRAINING_RECOVERY_FORMAT,
        "generation": generation,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "training_complete": training_complete,
        "bytes": byte_count,
        "sha256": sha256_file(target),
    }


def load_training_recovery(
    path: str,
    device: str | torch.device,
    *,
    expected_config_hash: str,
    expected_manifest_hash: str,
    expected_objective_config_sha256: str,
    expected_data_contract_sha256: str | None = None,
) -> dict[str, object]:
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
    loaded: object = torch.load(
        os.path.abspath(os.fspath(path)),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(loaded, dict):
        raise ValueError("training recovery checkpoint must be an object")
    payload = _object_dict(cast(object, loaded), "training recovery checkpoint")
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
        "objective",
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
        objective = ObjectiveConfig.from_document(payload["objective"])
        contract_objective = objective_from_ml_contract(payload["ml_contract"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training recovery objective configuration is invalid"
        ) from exc
    if train_config is None or (
        payload["objective_config_sha256"]
        != expected_objective_config_sha256
        or payload["objective_config_sha256"]
        != objective_config_sha256(objective)
        or contract_objective != objective
    ):
        raise ValueError(
            "training recovery objective configuration does not match the job"
        )
    if expected_data_contract_sha256 is not None:
        data_contract = _object_dict(
            payload["data_contract"],
            "training recovery data contract",
        )
        if (
            data_contract.get("dataContractSha256")
            != expected_data_contract_sha256
        ):
            raise ValueError(
                "training recovery data contract does not match the job"
            )
    generation = _positive(payload["generation"], "generation")
    completed_epochs = _positive(
        payload["completed_epochs"],
        "completed_epochs",
    )
    _nonnegative(payload["global_step"], "global_step")
    if generation != completed_epochs:
        raise ValueError(
            "training recovery generation must equal completed epochs"
        )
    if not isinstance(payload["training_complete"], bool):
        raise ValueError(
            "training recovery completion marker must be a boolean"
        )
    state = _object_dict(
        payload["trainer_state"],
        "training recovery trainer state",
    )
    progress = _object_dict(
        state.get("training_state"),
        "training recovery progress",
    )
    if (
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


def _target_schema_id(data_contract: Mapping[str, object] | None) -> str:
    if data_contract is None:
        return TARGET_SCHEMA_ID
    value = data_contract.get("targetSchemaId", data_contract.get("target_schema_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("training recovery target schema is unavailable")
    return value


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(
            lambda: source.read(_HASH_CHUNK_BYTES),
            b"",
        ):
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


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


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


def _object_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


__all__ = [
    "TRAINING_RECOVERY_FORMAT",
    "load_training_recovery",
    "save_training_recovery",
    "sha256_file",
]
