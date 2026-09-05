from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v12.constants import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    MAX_EVENT_BYTES,
)


class WorkerContractError(ValueError):
    """A worker document does not satisfy process contract v12."""


_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
_CONTRACTS_DIRECTORY = Path(__file__).parents[2]
_SCHEMA_DIRECTORIES = (
    _SCHEMA_DIRECTORY,
    _CONTRACTS_DIRECTORY / "semantic" / "v1" / "schemas",
    _CONTRACTS_DIRECTORY / "flight" / "v11" / "schemas",
    _CONTRACTS_DIRECTORY / "checkpoint" / "v6" / "schemas",
)
_SCHEMA_NAMES = frozenset({
    "arrow-manifest",
    "capabilities",
    "command-manifest",
    "control-message",
    "event",
    "prediction-manifest",
    "recovery-descriptor",
    "result-manifest",
    "training-metrics",
})


def validate_document(document: object, schema_name: str) -> JsonObject:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown worker contract schema: {schema_name}")
    if not isinstance(document, dict):
        raise WorkerContractError("worker contract document must be a JSON object")
    typed_document = cast(JsonObject, document)
    errors = sorted(
        _validator(schema_name).iter_errors(typed_document),
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
            document: object = json.load(
                source,
                object_pairs_hook=_unique_object,
            )
    except (OSError, UnicodeError, ValueError) as exc:
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
    return _encode_envelope(
        job_id=job_id,
        attempt=attempt,
        attempt_id=attempt_id,
        sequence=sequence,
        message_type=event_type,
        payload=payload,
        schema_name="event",
    )


def parse_event(line: bytes) -> JsonObject:
    return _parse_line(line, "event")


def encode_control_message(
    *,
    job_id: str,
    attempt: int,
    attempt_id: str,
    sequence: int,
    message_type: str,
    payload: JsonObject,
) -> bytes:
    return _encode_envelope(
        job_id=job_id,
        attempt=attempt,
        attempt_id=attempt_id,
        sequence=sequence,
        message_type=message_type,
        payload=payload,
        schema_name="control-message",
    )


def parse_control_message(line: bytes) -> JsonObject:
    return _parse_line(line, "control-message")


def _encode_envelope(
    *,
    job_id: str,
    attempt: int,
    attempt_id: str,
    sequence: int,
    message_type: str,
    payload: JsonObject,
    schema_name: str,
) -> bytes:
    document: JsonObject = {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "jobId": _canonical_uuid(job_id, "jobId"),
        "attempt": attempt,
        "attemptId": _canonical_uuid(attempt_id, "attemptId"),
        "sequence": sequence,
        "type": message_type,
        "payload": payload,
    }
    validate_document(document, schema_name)
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise WorkerContractError("worker message exceeds the size limit")
    return encoded + b"\n"


def _parse_line(line: bytes, schema_name: str) -> JsonObject:
    if not line.endswith(b"\n"):
        raise WorkerContractError("worker message is not newline terminated")
    if len(line) > MAX_EVENT_BYTES + 1:
        raise WorkerContractError("worker message exceeds the size limit")
    try:
        document: object = json.loads(
            line,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkerContractError("worker message is not valid UTF-8 JSON") from exc
    return validate_document(document, schema_name)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise WorkerContractError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise WorkerContractError(f"{label} must be a canonical UUID")
    return str(parsed)


def _validate_training_metric_invariants(document: JsonObject) -> None:
    direct = _target_series(document, "directLosses")
    metrics = _target_series(document, "targetMetrics")
    if direct != metrics:
        raise WorkerContractError("training target series differ")

    batches = _integer(document, "batches")
    completed = _integer(document, "trainingBatchesCompleted")
    applied = _integer(document, "optimizerUpdatesApplied")
    skipped = _integer(document, "optimizerUpdatesSkipped")
    overflows = _integer(document, "ampOverflowBatches")
    finite = _integer(document, "finiteGradientBatches")
    non_finite = _integer(document, "nonFiniteGradientBatches")
    if completed != batches:
        raise WorkerContractError("trainingBatchesCompleted differs from batches")
    if completed != applied + skipped:
        raise WorkerContractError(
            "optimizer update counts differ from completed batches"
        )
    if completed != finite + non_finite:
        raise WorkerContractError("gradient counts differ from completed batches")
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
        _number(value, "gradient statistic") for value in statistics
    )
    if mean > maximum or percentile > maximum:
        raise WorkerContractError("gradient statistics are inconsistent")


def _target_series(document: JsonObject, field: str) -> tuple[str, ...]:
    value = document[field]
    if not isinstance(value, list):
        raise WorkerContractError(f"{field} must be an array")
    identities: list[str] = []
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, Mapping):
            raise WorkerContractError(f"{field}[{index}] must be an object")
        mapping = cast(Mapping[object, object], item)
        identity = mapping.get("targetIdentity")
        if mapping.get("targetIndex") != index or not isinstance(identity, str):
            raise WorkerContractError(
                f"{field}[{index}] target identity is inconsistent"
            )
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise WorkerContractError(f"{field} target identities must be unique")
    return tuple(identities)


def _integer(document: JsonObject, field: str) -> int:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerContractError(f"{field} must be an integer")
    return value


def _number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerContractError(f"{label} must be a number")
    return float(value)


@cache
def _validator(schema_name: str) -> Validator:
    schema = _load_schema(_SCHEMA_DIRECTORY / f"{schema_name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )


@cache
def _schema_registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for directory in _SCHEMA_DIRECTORIES:
        for path in sorted(directory.glob("*.schema.json")):
            contents = _load_schema(path)
            if isinstance(contents, bool):
                raise WorkerContractError("worker schema must be an object")
            mapping = cast(Mapping[str, object], contents)
            schema_id = mapping.get("$id")
            if not isinstance(schema_id, str):
                raise WorkerContractError("worker schema has no string $id")
            registry = registry.with_resource(
                schema_id,
                Resource.from_contents(contents),
            )
    return registry


def _load_schema(path: Path) -> Schema:
    with open(path, encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, Mapping):
        raise WorkerContractError("worker schema must be a JSON object")
    return cast(Schema, document)


__all__ = [
    "WorkerContractError",
    "encode_control_message",
    "encode_event",
    "load_document",
    "parse_control_message",
    "parse_event",
    "validate_document",
]
