from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACT_NAME = "transformer-worker"
CONTRACT_VERSION = 3
MAX_EVENT_BYTES = 1024 * 1024

_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
_SCHEMA_NAMES = frozenset({
    "arrow-manifest",
    "capabilities",
    "command-manifest",
    "control-message",
    "event",
    "result-manifest",
})


class WorkerContractError(ValueError):
    """A worker document does not satisfy the neutral process contract."""


def validate_document(document: Any, schema_name: str) -> dict[str, Any]:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown worker contract schema: {schema_name}")
    if not isinstance(document, dict):
        raise WorkerContractError("worker contract document must be a JSON object")

    schema = _load_schema(schema_name)
    registry = _schema_registry()
    validator = Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{location}: " if location else ""
        raise WorkerContractError(f"{prefix}{error.message}")
    return document


def load_document(path: str | os.PathLike[str], schema_name: str) -> dict[str, Any]:
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
    payload: dict[str, Any],
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


def parse_event(line: bytes) -> dict[str, Any]:
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
    payload: dict[str, Any],
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


def parse_control_message(line: bytes) -> dict[str, Any]:
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


def _load_schema(name: str) -> dict[str, Any]:
    with open(_SCHEMA_DIRECTORY / f"{name}.schema.json", encoding="utf-8") as source:
        return json.load(source)


def _schema_registry() -> Registry:
    registry = Registry()
    for path in sorted(_SCHEMA_DIRECTORY.glob("*.schema.json")):
        with open(path, encoding="utf-8") as source:
            contents = json.load(source)
        registry = registry.with_resource(
            contents["$id"],
            Resource.from_contents(contents),
        )
    return registry
