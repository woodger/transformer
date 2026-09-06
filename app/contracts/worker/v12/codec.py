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
from app.contracts.semantic.v1 import ModelContract
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
    _CONTRACTS_DIRECTORY / "flight" / "v12" / "schemas",
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


def validate_training_metrics_for_model(
    document: object,
    model_contract: ModelContract | object,
) -> JsonObject:
    metrics = validate_document(document, "training-metrics")
    contract = (
        model_contract
        if isinstance(model_contract, ModelContract)
        else ModelContract.from_document(model_contract)
    )
    slots = contract.target_identities
    direct = _object_items(metrics, "directLosses")
    target_metrics = _object_items(metrics, "targetMetrics")
    auxiliary = _object_items(metrics, "auxiliaryLosses")

    expected_direct = contract.direct_components
    if len(direct) != len(expected_direct):
        raise WorkerContractError("training direct losses differ from objective")
    for index, (actual, expected) in enumerate(
        zip(direct, expected_direct, strict=True)
    ):
        if (
            actual.get("componentIdentity") != expected.get("identity")
            or actual.get("operator") != expected.get("operator")
            or actual.get("targetIdentity") != slots[index]
            or actual.get("targetIndex") != index
        ):
            raise WorkerContractError(
                "training direct losses differ from objective"
            )

    if len(target_metrics) != len(slots) or any(
        item.get("targetIdentity") != identity
        or item.get("targetIndex") != index
        for index, (item, identity) in enumerate(
            zip(target_metrics, slots, strict=True)
        )
    ):
        raise WorkerContractError("training target metrics differ from target layout")

    expected_auxiliary = contract.auxiliary_components
    if len(auxiliary) != len(expected_auxiliary) or any(
        actual.get("componentIdentity") != expected.get("identity")
        or actual.get("operator") != expected.get("operator")
        for actual, expected in zip(
            auxiliary,
            expected_auxiliary,
            strict=True,
        )
    ):
        raise WorkerContractError(
            "training auxiliary losses differ from objective"
        )

    interactions = metrics["gradientInteractions"]
    if interactions is not None:
        _validate_gradient_interactions(
            cast(JsonObject, interactions),
            contract,
        )
    return metrics


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


def _object_items(document: JsonObject, field: str) -> tuple[JsonObject, ...]:
    value = document[field]
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise WorkerContractError(f"{field} must contain objects")
    return tuple(cast(JsonObject, item) for item in value)


def _validate_gradient_interactions(
    document: JsonObject,
    contract: ModelContract,
) -> None:
    components = _object_items(document, "components")
    pairs = _object_items(document, "pairs")
    direct_targets = {
        str(component["identity"]): (index, contract.target_identities[index])
        for index, component in enumerate(contract.direct_components)
    }
    objective_identities = {
        str(component["identity"])
        for component in (
            *contract.direct_components,
            *contract.auxiliary_components,
        )
    }
    seen_components: set[str] = set()
    for component in components:
        identity = cast(str, component["componentIdentity"])
        if identity not in objective_identities or identity in seen_components:
            raise WorkerContractError(
                "gradient component does not resolve to the objective"
            )
        seen_components.add(identity)
        expected_target = direct_targets.get(identity)
        actual_target = (
            component.get("targetIndex"),
            component.get("targetIdentity"),
        )
        if (
            expected_target is None
            and actual_target != (None, None)
            or expected_target is not None
            and actual_target != expected_target
        ):
            raise WorkerContractError(
                "gradient component target differs from the objective"
            )

    seen_pairs: set[tuple[str, str]] = set()
    for pair in pairs:
        identity = (
            cast(str, pair["leftComponentIdentity"]),
            cast(str, pair["rightComponentIdentity"]),
        )
        if (
            identity[0] == identity[1]
            or identity[0] not in objective_identities
            or identity[1] not in objective_identities
            or identity in seen_pairs
        ):
            raise WorkerContractError(
                "gradient pair does not resolve to distinct objective components"
            )
        seen_pairs.add(identity)


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
    "validate_training_metrics_for_model",
]
