from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v7 import CONTRACT_NAME, CONTRACT_VERSION
from app.contracts.worker.v7.config import (
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.contracts.worker.v7.objective import (
    CHECKPOINT_FORMAT,
    ml_contract,
    objective_config,
)
from app.worker.application.documents import integer_field, object_field, string_field
from app.worker.runtime.version import __version__
from app.worker.training.trainer import Trainer

_COPY_CHUNK_BYTES = 1024 * 1024


class CommittedInputArtifacts:
    """Verify each immutable input receipt exactly once per worker attempt."""

    def __init__(self) -> None:
        self._verified: dict[
            int,
            tuple[tuple[object, ...], str],
        ] = {}

    def path(self, item: JsonObject) -> str:
        ordinal = integer_field(item, "ordinal")
        identity = _input_receipt_identity(item)
        existing = self._verified.get(ordinal)
        if existing is None:
            path = validate_artifact(object_field(item, "artifact"))
            self._verified[ordinal] = (identity, path)
            return path
        if existing[0] != identity:
            raise ValueError(
                "committed input receipt changed during the worker attempt"
            )
        return existing[1]


def result_identity(manifest: JsonObject) -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "jobId": string_field(manifest, "jobId"),
        "attempt": integer_field(manifest, "attempt"),
        "attemptId": string_field(manifest, "attemptId"),
        "operation": string_field(manifest, "operation"),
    }


def checkpoint_metadata(
    trainer: Trainer,
    data_contract: JsonObject | None = None,
) -> JsonObject:
    model_config = trainer.model_config
    train_config = trainer.train_config
    if model_config is None:
        raise ValueError("fit checkpoint configuration is unavailable")
    feature_dim = model_config.feature_dim
    if feature_dim is None:
        raise ValueError("fit checkpoint feature dimension is unavailable")
    if data_contract is None:
        raise ValueError("fit checkpoint data contract is unavailable")
    best_selection_score = trainer.best_selection_score
    if not math.isfinite(best_selection_score):
        best_selection_score = None
    selection_enabled = trainer.selection is not None
    return {
        "format": CHECKPOINT_FORMAT,
        "serviceVersion": __version__,
        "modelConfig": model_config_to_manifest(model_config),
        "trainingConfig": train_config_to_manifest(train_config),
        "dataContract": dict(data_contract),
        "mlContract": ml_contract(train_config),
        "objectiveConfig": objective_config(train_config),
        "checkpointSelection": {
            "enabled": selection_enabled,
            "objectiveConfigSha256": ml_contract(train_config)[
                "objectiveConfigSha256"
            ],
            "bestSelectionScore": best_selection_score,
            "bestFrame": trainer.best_frame,
            "bestEpoch": trainer.best_epoch,
            "source": (
                "best_selection_score"
                if selection_enabled
                else "last_maximum_stage"
            ),
        },
    }


def validate_workspace(path: str) -> str:
    workspace = os.path.abspath(os.fspath(path))
    if not os.path.isabs(path) or os.path.realpath(workspace) != workspace:
        raise ValueError("worker workspace must be a canonical managed path")
    if not os.path.isdir(workspace):
        raise ValueError("worker workspace is unavailable")
    return workspace


def validate_artifact(document: JsonObject) -> str:
    documented_path = string_field(document, "path")
    path = os.path.abspath(documented_path)
    if not os.path.isabs(documented_path):
        raise ValueError("worker artifact path must be absolute")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ValueError("worker input artifact is unavailable") from exc
    if (
        size != integer_field(document, "byteCount")
        or _sha256_file(path) != string_field(document, "sha256")
    ):
        raise ValueError("worker input artifact integrity check failed")
    return path


def artifact_document(path: str) -> JsonObject:
    path = os.path.abspath(path)
    return {
        "path": path,
        "byteCount": os.path.getsize(path),
        "sha256": _sha256_file(path),
    }


def write_json_once(path: str, document: JsonObject) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=parent,
        prefix=f".{Path(path).name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("worker result manifest already exists") from exc
        _fsync_directory(parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("worker JSON field names must be strings")
        return {
            cast(str, key): json_safe(item)
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            json_safe(item)
            for item in cast(Sequence[object], value)
        ]
    raise TypeError(
        f"worker value is not JSON-compatible: {type(value).__name__}"
    )


def boolean_value(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"worker {label} must be a boolean")
    return value


def _input_receipt_identity(item: JsonObject) -> tuple[object, ...]:
    artifact = object_field(item, "artifact")
    return (
        string_field(item, "schemaId"),
        integer_field(item, "ordinal"),
        integer_field(item, "commitRevision"),
        string_field(item, "dataContractSha256"),
        integer_field(item, "rows"),
        string_field(artifact, "path"),
        integer_field(artifact, "byteCount"),
        string_field(artifact, "sha256"),
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: str) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CommittedInputArtifacts",
    "artifact_document",
    "boolean_value",
    "checkpoint_metadata",
    "json_safe",
    "result_identity",
    "validate_artifact",
    "validate_workspace",
    "write_json_once",
]
