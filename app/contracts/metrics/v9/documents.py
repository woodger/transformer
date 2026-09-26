from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonObject, JsonValue

ARTIFACT_FORMAT = "transformer.training-metrics.v9"
ARTIFACT_MEDIA_TYPE = "application/x-ndjson"
POINT_DOCUMENT_SCHEMA = "transformer.metrics-point.v9"
POINT_INDEX = "metrics-points-v9"

_ROOT = Path(__file__).parent
_SEMANTIC_SCHEMAS = _ROOT.parents[1] / "semantic" / "v4" / "schemas"


def build_training_record(
    metrics: JsonObject,
    *,
    recorded_at: float,
    job_id: str,
    attempt_id: str,
    attempt: int,
    model_ref: str,
    semantic_digests: JsonObject,
    checkpoint_format: str,
    application_version: str,
    git_commit: str,
    targets: Sequence[str],
) -> JsonObject:
    document: JsonObject = {
        "format": ARTIFACT_FORMAT,
        "recordedAt": _utc_timestamp(recorded_at),
        "jobId": job_id,
        "attemptId": attempt_id,
        "attempt": attempt,
        "modelRef": model_ref,
        "semanticDigests": dict(semantic_digests),
        "checkpointFormat": checkpoint_format,
        "transformerVersion": application_version,
        "transformerGitCommit": git_commit,
        "targets": [
            {"index": index, "identity": identity}
            for index, identity in enumerate(targets)
        ],
        **metrics,
    }
    return validate_training_record(document)


def validate_training_record(document: object) -> JsonObject:
    typed = _validate(document, "training-record", "training metrics record")
    targets = _target_layout(typed, "targets")
    if _target_series(typed, "directLosses") != targets:
        raise ValueError("invalid training metrics record: direct targets differ")
    if _target_series(typed, "targetMetrics") != targets:
        raise ValueError("invalid training metrics record: target metrics differ")
    _validate_batch_counts(typed)
    return typed


def validate_point_document(document: object) -> JsonObject:
    return _validate(document, "point", "metrics point")


def project_training_points(
    record: JsonObject,
    *,
    deployment_id: str,
) -> tuple[JsonObject, ...]:
    record = validate_training_record(record)
    points: list[JsonObject] = []
    for field, name, unit in _SCALAR_METRICS:
        value = record[field]
        if value is not None:
            points.append(
                _point(
                    record,
                    deployment_id=deployment_id,
                    name=name,
                    value=cast(JsonValue, value),
                    unit=unit,
                )
            )
    for item in _objects(record, "directLosses"):
        points.append(
            _point(
                record,
                deployment_id=deployment_id,
                name="training.loss.direct",
                value=item["value"],
                unit="1",
                target={
                    "identity": item["targetIdentity"],
                    "index": item["targetIndex"],
                },
                component={
                    "identity": item["componentIdentity"],
                    "operator": item["operator"],
                },
            )
        )
    for item in _objects(record, "auxiliaryLosses"):
        points.append(
            _point(
                record,
                deployment_id=deployment_id,
                name="training.loss.auxiliary",
                value=item["value"],
                unit="1",
                component={
                    "identity": item["componentIdentity"],
                    "operator": item["operator"],
                },
            )
        )
    for item in _objects(record, "targetMetrics"):
        target: JsonObject = {
            "identity": item["targetIdentity"],
            "index": item["targetIndex"],
        }
        for field, name in (
            ("mae", "training.target.mae"),
            ("rmse", "training.target.rmse"),
        ):
            points.append(
                _point(
                    record,
                    deployment_id=deployment_id,
                    name=name,
                    value=item[field],
                    unit="1",
                    target=target,
                )
            )
    interactions = record["gradientInteractions"]
    if isinstance(interactions, dict):
        for item in _objects(cast(JsonObject, interactions), "components"):
            gradient_target: JsonObject | None = None
            if "targetIdentity" in item:
                gradient_target = {
                    "identity": item["targetIdentity"],
                    "index": item["targetIndex"],
                }
            points.append(
                _point(
                    record,
                    deployment_id=deployment_id,
                    name="training.gradient.component.norm",
                    value=item["meanNorm"],
                    unit="1",
                    target=gradient_target,
                    component={
                        "identity": item["componentIdentity"],
                        "operator": _component_operator(record, item),
                    },
                )
            )
        for item in _objects(cast(JsonObject, interactions), "pairs"):
            pair: JsonObject = {
                "leftComponentIdentity": item["leftComponentIdentity"],
                "rightComponentIdentity": item["rightComponentIdentity"],
            }
            for field, name in (
                ("meanCosine", "training.gradient.pair.cosine"),
                ("negativeCosineFraction", "training.gradient.pair.negative_fraction"),
            ):
                points.append(
                    _point(
                        record,
                        deployment_id=deployment_id,
                        name=name,
                        value=item[field],
                        unit="ratio" if field == "negativeCosineFraction" else "1",
                        pair=pair,
                    )
                )
    return tuple(points)


def _point(
    record: JsonObject,
    *,
    deployment_id: str,
    name: str,
    value: JsonValue,
    unit: str,
    target: JsonObject | None = None,
    component: JsonObject | None = None,
    pair: JsonObject | None = None,
) -> JsonObject:
    context: JsonObject = {}
    if target is not None:
        context["target"] = dict(target)
    if component is not None:
        context["component"] = dict(component)
    if pair is not None:
        context["pair"] = dict(pair)
    identity: list[JsonValue] = [
        POINT_DOCUMENT_SCHEMA,
        deployment_id,
        cast(str, record["jobId"]),
        cast(str, record["attemptId"]),
        record["frame"],
        cast(int, record["epoch"]),
        cast(int, record["step"]),
        name,
        cast(JsonValue, target),
        cast(JsonValue, component),
        cast(JsonValue, pair),
    ]
    document: JsonObject = {
        "schema": POINT_DOCUMENT_SCHEMA,
        "eventId": _jcs_sha256(identity),
        "recordedAt": record["recordedAt"],
        "deploymentId": deployment_id,
        "runId": record["jobId"],
        "attemptId": record["attemptId"],
        "attempt": record["attempt"],
        "modelRef": record["modelRef"],
        "semanticDigests": record["semanticDigests"],
        "epoch": record["epoch"],
        "step": record["step"],
        **context,
        "metric": {"name": name, "value": value, "unit": unit},
    }
    document["documentSha256"] = _jcs_sha256(document)
    return validate_point_document(document)


def _component_operator(record: JsonObject, component: JsonObject) -> str:
    identity = component["componentIdentity"]
    for item in (
        *_objects(record, "directLosses"),
        *_objects(record, "auxiliaryLosses"),
    ):
        if item["componentIdentity"] == identity:
            return cast(str, item["operator"])
    raise ValueError("gradient component does not resolve to an objective component")


def _target_layout(document: JsonObject, field: str) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(_objects(document, field)):
        if item.get("index") != index or not isinstance(item.get("identity"), str):
            raise ValueError(f"invalid training metrics record: {field} order differs")
        result.append(cast(str, item["identity"]))
    if len(result) != len(set(result)):
        raise ValueError(f"invalid training metrics record: {field} identities differ")
    return tuple(result)


def _target_series(document: JsonObject, field: str) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(_objects(document, field)):
        if item.get("targetIndex") != index or not isinstance(
            item.get("targetIdentity"), str
        ):
            raise ValueError(f"invalid training metrics record: {field} order differs")
        result.append(cast(str, item["targetIdentity"]))
    return tuple(result)


def _validate_batch_counts(document: JsonObject) -> None:
    batches = _integer(document, "batches")
    completed = _integer(document, "trainingBatchesCompleted")
    applied = _integer(document, "optimizerUpdatesApplied")
    skipped = _integer(document, "optimizerUpdatesSkipped")
    finite = _integer(document, "finiteGradientBatches")
    non_finite = _integer(document, "nonFiniteGradientBatches")
    overflow = _integer(document, "ampOverflowBatches")
    if completed != batches or completed != applied + skipped:
        raise ValueError("invalid training metrics record: optimizer counts differ")
    if completed != finite + non_finite or overflow > skipped or overflow > non_finite:
        raise ValueError("invalid training metrics record: gradient counts differ")
    statistics = tuple(
        document[field]
        for field in (
            "preClipGradientNormMean",
            "preClipGradientNormMax",
            "preClipGradientNormP95",
        )
    )
    if finite == 0 and any(value is not None for value in statistics):
        raise ValueError("invalid training metrics record: gradient statistics differ")
    if finite > 0 and any(value is None for value in statistics):
        raise ValueError("invalid training metrics record: gradient statistics differ")


def _validate(document: object, schema: str, label: str) -> JsonObject:
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be an object")
    typed = cast(JsonObject, document)
    errors = sorted(
        _validator(schema).iter_errors(typed),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{location}: " if location else ""
        raise ValueError(f"invalid {label}: {prefix}{error.message}")
    _finite(typed, label)
    return typed


@cache
def _validator(name: str) -> Validator:
    schema = _load_schema(_ROOT / f"{name}.schema.json")
    return Draft202012Validator(
        schema,
        registry=_registry(),
        format_checker=FormatChecker(),
    )


@cache
def _registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for directory in (_ROOT, _SEMANTIC_SCHEMAS):
        for path in sorted(directory.glob("*.schema.json")):
            schema = _load_schema(path)
            schema_id = cast(Mapping[str, object], schema).get("$id")
            if isinstance(schema_id, str):
                registry = registry.with_resource(
                    schema_id, Resource.from_contents(schema)
                )
    return registry


def _load_schema(path: Path) -> Schema:
    with open(path, encoding="utf-8") as source:
        return cast(Schema, json.load(source))


def _finite(value: JsonValue, label: str) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _finite(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _finite(nested, label)
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(value)
    ):
        raise ValueError(f"invalid {label}: numeric values must be finite")


def _objects(document: JsonObject, field: str) -> tuple[JsonObject, ...]:
    value = document[field]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"metrics field {field} must contain objects")
    return tuple(cast(JsonObject, item) for item in value)


def _integer(document: JsonObject, field: str) -> int:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metrics field {field} must be an integer")
    return value


def _jcs_sha256(value: JsonValue) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _utc_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


_SCALAR_METRICS = (
    ("loss", "training.loss.total", "1"),
    ("selectionScore", "training.selection.score", "1"),
    ("trainingBatchesCompleted", "training.batches.completed", "count"),
    ("optimizerUpdatesApplied", "training.optimizer.updates.applied", "count"),
    ("optimizerUpdatesSkipped", "training.optimizer.updates.skipped", "count"),
    ("ampOverflowBatches", "training.amp.overflow_batches", "count"),
    ("finiteGradientBatches", "training.gradient.batches.finite", "count"),
    ("nonFiniteGradientBatches", "training.gradient.batches.non_finite", "count"),
    ("preClipGradientNormMean", "training.gradient.pre_clip_norm.mean", "1"),
    ("preClipGradientNormMax", "training.gradient.pre_clip_norm.max", "1"),
    ("preClipGradientNormP95", "training.gradient.pre_clip_norm.p95", "1"),
    ("lr", "training.learning_rate", "1"),
    ("nanRatio", "training.input.nan_ratio", "ratio"),
    ("maskedTokenRatio", "training.input.masked_token_ratio", "ratio"),
    ("completeTokenRatio", "training.input.complete_token_ratio", "ratio"),
    ("partialTokenRatio", "training.input.partial_token_ratio", "ratio"),
    ("emptyTokenRatio", "training.input.empty_token_ratio", "ratio"),
    ("inputPipelineMs", "training.phase.input_pipeline", "ms"),
    ("missingStatsMs", "training.phase.missing_stats", "ms"),
    ("hostToDeviceMs", "training.phase.host_to_device", "ms"),
    ("trainStepMs", "training.phase.train_step", "ms"),
    ("elapsedMs", "training.phase.epoch_elapsed", "ms"),
)


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_MEDIA_TYPE",
    "POINT_INDEX",
    "build_training_record",
    "project_training_points",
    "validate_point_document",
    "validate_training_record",
]
