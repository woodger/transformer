from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonObject
from app.contracts.ml import canonical_targets

CONTRACT_NAME = "transformer-worker"
CONTRACT_VERSION = 11
MAX_EVENT_BYTES = 1024 * 1024

_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
_SCHEMA_NAMES = frozenset({
    "arrow-manifest",
    "capabilities",
    "command-manifest",
    "control-message",
    "event",
    "prediction-manifest",
    "result-manifest",
    "training-metrics",
})


class WorkerContractError(ValueError):
    """A worker document does not satisfy the neutral process contract."""


def validate_document(document: object, schema_name: str) -> JsonObject:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown worker contract schema: {schema_name}")
    if not isinstance(document, dict):
        raise WorkerContractError("worker contract document must be a JSON object")
    typed_document = cast(JsonObject, document)

    schema = _load_schema(schema_name)
    registry = _schema_registry()
    validator = Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    # jsonschema's public validator overload exposes its yielded validation
    # errors as Any; schema validation below is the runtime trust boundary.
    errors = sorted(
        validator.iter_errors(  # pyright: ignore[reportUnknownMemberType]
            typed_document
        ),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{location}: " if location else ""
        raise WorkerContractError(f"{prefix}{error.message}")
    if schema_name == "training-metrics":
        _validate_training_metric_invariants(typed_document)
    return typed_document


def load_document(path: str | os.PathLike[str], schema_name: str) -> JsonObject:
    try:
        with open(path, encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerContractError(
            f"worker {schema_name} document could not be read"
        ) from exc
    return validate_document(document, schema_name)


def encode_event(
    *,
    job_id: str,
    attempt: int,
    attempt_id: str,
    sequence: int,
    event_type: str,
    payload: JsonObject,
) -> bytes:
    document = {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "jobId": _canonical_uuid(job_id, "jobId"),
        "attempt": attempt,
        "attemptId": _canonical_uuid(attempt_id, "attemptId"),
        "sequence": sequence,
        "type": event_type,
        "payload": payload,
    }
    validate_document(document, "event")
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise WorkerContractError("worker event exceeds the size limit")
    return encoded + b"\n"


def parse_event(line: bytes) -> JsonObject:
    if not line.endswith(b"\n"):
        raise WorkerContractError("worker event is not newline terminated")
    if len(line) > MAX_EVENT_BYTES + 1:
        raise WorkerContractError("worker event exceeds the size limit")
    try:
        document = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerContractError("worker event is not valid UTF-8 JSON") from exc
    return validate_document(document, "event")


def encode_control_message(
    *,
    job_id: str,
    attempt: int,
    attempt_id: str,
    sequence: int,
    message_type: str,
    payload: JsonObject,
) -> bytes:
    document = {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "jobId": _canonical_uuid(job_id, "jobId"),
        "attempt": attempt,
        "attemptId": _canonical_uuid(attempt_id, "attemptId"),
        "sequence": sequence,
        "type": message_type,
        "payload": payload,
    }
    validate_document(document, "control-message")
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise WorkerContractError("worker control message exceeds the size limit")
    return encoded + b"\n"


def parse_control_message(line: bytes) -> JsonObject:
    if not line.endswith(b"\n"):
        raise WorkerContractError("worker control message is not newline terminated")
    if len(line) > MAX_EVENT_BYTES + 1:
        raise WorkerContractError("worker control message exceeds the size limit")
    try:
        document = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerContractError(
            "worker control message is not valid UTF-8 JSON"
        ) from exc
    return validate_document(document, "control-message")


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise WorkerContractError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise WorkerContractError(f"{label} must be a canonical UUID")
    return str(parsed)


def _validate_training_metric_invariants(document: JsonObject) -> None:
    direct_targets = _validate_target_series(document, "directLosses")
    metric_targets = _validate_target_series(document, "targetMetrics")
    if direct_targets != metric_targets:
        raise WorkerContractError("training target series differ")
    batches = _metric_integer(document, "batches")
    completed = _metric_integer(document, "trainingBatchesCompleted")
    applied = _metric_integer(document, "optimizerUpdatesApplied")
    skipped = _metric_integer(document, "optimizerUpdatesSkipped")
    overflows = _metric_integer(document, "ampOverflowBatches")
    finite = _metric_integer(document, "finiteGradientBatches")
    non_finite = _metric_integer(document, "nonFiniteGradientBatches")
    if completed != batches:
        raise WorkerContractError(
            "trainingBatchesCompleted differs from batches"
        )
    if completed != applied + skipped:
        raise WorkerContractError(
            "optimizer update counts differ from completed batches"
        )
    if completed != finite + non_finite:
        raise WorkerContractError(
            "gradient counts differ from completed batches"
        )
    if overflows > skipped or overflows > non_finite:
        raise WorkerContractError("AMP overflow counts are inconsistent")
    statistics = tuple(
        document[field]
        for field in (
            "preClipGradientNormMean",
            "preClipGradientNormMax",
            "preClipGradientNormP95",
        )
    )
    if finite == 0:
        if any(value is not None for value in statistics):
            raise WorkerContractError(
                "gradient statistics require a finite gradient batch"
            )
        return
    if any(value is None for value in statistics):
        raise WorkerContractError("gradient statistics are incomplete")
    mean, maximum, percentile = (
        _metric_number(value, "gradient statistic")
        for value in statistics
    )
    if mean > maximum or percentile > maximum:
        raise WorkerContractError("gradient statistics are inconsistent")


def _validate_target_series(
    document: JsonObject,
    field: str,
) -> tuple[str, ...]:
    value = document[field]
    if not isinstance(value, list):
        raise WorkerContractError(f"{field} must be an array")
    names: list[str] = []
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, dict):
            raise WorkerContractError(f"{field}[{index}] must be an object")
        target = cast(dict[object, object], item).get("target")
        if not isinstance(target, dict):
            raise WorkerContractError(
                f"{field}[{index}].target must be an object"
            )
        target_mapping = cast(dict[object, object], target)
        name = target_mapping.get("name")
        if target_mapping.get("index") != index or not isinstance(name, str):
            raise WorkerContractError(
                f"{field}[{index}] target identity is inconsistent"
            )
        names.append(name)
    try:
        return canonical_targets(names)
    except ValueError as exc:
        raise WorkerContractError(f"{field} target selection is invalid") from exc


def _metric_integer(document: JsonObject, field: str) -> int:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerContractError(f"{field} must be an integer")
    return value


def _metric_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerContractError(f"{label} must be a number")
    return float(value)


def _load_schema(name: str) -> Schema:
    with open(_SCHEMA_DIRECTORY / f"{name}.schema.json", encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, dict):
        raise WorkerContractError("worker schema must be a JSON object")
    return cast(Schema, document)


def _schema_registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for path in sorted(_SCHEMA_DIRECTORY.glob("*.schema.json")):
        with open(path, encoding="utf-8") as source:
            document: object = json.load(source)
        if not isinstance(document, dict):
            raise WorkerContractError("worker schema must be a JSON object")
        contents = cast(Schema, document)
        if isinstance(contents, bool):
            raise WorkerContractError("worker schema must be a JSON object")
        schema_id = contents.get("$id")
        if not isinstance(schema_id, str):
            raise WorkerContractError("worker schema must define a string $id")
        registry = registry.with_resource(
            schema_id,
            Resource.from_contents(contents),
        )
    return registry
