from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.ml import canonical_targets

ARTIFACT_FORMAT = "transformer.training-metrics.v4"
ARTIFACT_MEDIA_TYPE = "application/x-ndjson"
PROJECTION_VERSION = "inventory.metrics.v6"
POINT_DOCUMENT_SCHEMA = "inventory.metrics.point.v4"
POINT_INDEX = "metrics-points-v4"

_SCHEMA_PATH = Path(__file__).with_name("training-record.schema.json")
_TRAINING_RECORD_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_TRAINING_RECORD_VALIDATOR = Draft202012Validator(
    _TRAINING_RECORD_SCHEMA,
    format_checker=FormatChecker(),
)
_POINT_VALIDATOR = Draft202012Validator(
    json.loads(Path(__file__).with_name("point.schema.json").read_text(
        encoding="utf-8"
    )),
    format_checker=FormatChecker(),
)


@dataclass(frozen=True, slots=True)
class _Metric:
    field: str
    name: str
    unit: str = "1"


_METRICS = (
    _Metric("loss", "training.loss.total"),
    _Metric("selection_score", "training.selection.score"),
    _Metric(
        "trainingBatchesCompleted",
        "training.batches.completed",
        "count",
    ),
    _Metric(
        "optimizerUpdatesApplied",
        "training.optimizer.updates.applied",
        "count",
    ),
    _Metric(
        "optimizerUpdatesSkipped",
        "training.optimizer.updates.skipped",
        "count",
    ),
    _Metric(
        "ampOverflowBatches",
        "training.amp.overflow_batches",
        "count",
    ),
    _Metric(
        "finiteGradientBatches",
        "training.gradient.batches.finite",
        "count",
    ),
    _Metric(
        "nonFiniteGradientBatches",
        "training.gradient.batches.non_finite",
        "count",
    ),
    _Metric(
        "preClipGradientNormMean",
        "training.gradient.pre_clip_norm.mean",
    ),
    _Metric(
        "preClipGradientNormMax",
        "training.gradient.pre_clip_norm.max",
    ),
    _Metric(
        "preClipGradientNormP95",
        "training.gradient.pre_clip_norm.p95",
    ),
    _Metric("lr", "training.learning_rate"),
    _Metric("nan_ratio", "training.input.nan_ratio", "ratio"),
    _Metric(
        "masked_token_ratio",
        "training.input.masked_token_ratio",
        "ratio",
    ),
    _Metric(
        "complete_token_ratio",
        "training.input.complete_token_ratio",
        "ratio",
    ),
    _Metric(
        "partial_token_ratio",
        "training.input.partial_token_ratio",
        "ratio",
    ),
    _Metric("empty_token_ratio", "training.input.empty_token_ratio", "ratio"),
    _Metric(
        "input_pipeline_ms",
        "training.phase.input_pipeline",
        "ms",
    ),
    _Metric("missing_stats_ms", "training.phase.missing_stats", "ms"),
    _Metric("host_to_device_ms", "training.phase.host_to_device", "ms"),
    _Metric("train_step_ms", "training.phase.train_step", "ms"),
    _Metric("elapsed_ms", "training.phase.epoch_elapsed", "ms"),
)


def build_training_record(
    metrics: JsonObject,
    *,
    recorded_at: float,
    job_id: str,
    attempt_id: str,
    attempt: int,
    model_ref: str,
    data_contract_sha256: str,
    objective_config_sha256: str,
    checkpoint_format: str,
    application_version: str,
    git_commit: str,
    targets: Sequence[str],
) -> JsonObject:
    selected = canonical_targets(targets)
    document: JsonObject = {
        "format": ARTIFACT_FORMAT,
        "recordedAt": _utc_timestamp(recorded_at),
        "jobId": job_id,
        "attemptId": attempt_id,
        "attempt": attempt,
        "modelRef": model_ref,
        "dataContractSha256": data_contract_sha256,
        "objectiveConfigSha256": objective_config_sha256,
        "checkpointFormat": checkpoint_format,
        "transformerVersion": application_version,
        "transformerGitCommit": git_commit,
        "targets": list(selected),
        **metrics,
    }
    return validate_training_record(document)


def validate_training_record(document: object) -> JsonObject:
    if not isinstance(document, dict):
        raise ValueError("training metrics record must be an object")
    mapping = cast(dict[object, object], document)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("training metrics record keys must be strings")
    typed_document = cast(JsonObject, document)
    errors = sorted(
        _TRAINING_RECORD_VALIDATOR.iter_errors(  # pyright: ignore[reportUnknownMemberType]
            typed_document
        ),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{location}: " if location else ""
        raise ValueError(f"invalid training metrics record: {prefix}{error.message}")
    _validate_finite_numbers(typed_document, name="training metrics record")
    _validate_training_metric_invariants(typed_document)
    return typed_document


def _validate_training_metric_invariants(document: JsonObject) -> None:
    selected = canonical_targets(_string_list(document, "targets"))
    direct_targets = tuple(
        _target_name(item) for item in _target_series(document, "directLosses")
    )
    metric_targets = tuple(
        _target_name(item) for item in _target_series(document, "targetMetrics")
    )
    if direct_targets != selected or metric_targets != selected:
        raise ValueError("invalid training metrics record: target series differ")
    batches = _integer(document, "batches")
    completed = _integer(document, "trainingBatchesCompleted")
    applied = _integer(document, "optimizerUpdatesApplied")
    skipped = _integer(document, "optimizerUpdatesSkipped")
    overflows = _integer(document, "ampOverflowBatches")
    finite = _integer(document, "finiteGradientBatches")
    non_finite = _integer(document, "nonFiniteGradientBatches")
    if completed != batches:
        raise ValueError(
            "invalid training metrics record: completed batches differ"
        )
    if completed != applied + skipped:
        raise ValueError(
            "invalid training metrics record: optimizer counts differ"
        )
    if completed != finite + non_finite:
        raise ValueError(
            "invalid training metrics record: gradient counts differ"
        )
    if overflows > skipped or overflows > non_finite:
        raise ValueError(
            "invalid training metrics record: AMP overflow counts differ"
        )
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
            raise ValueError(
                "invalid training metrics record: gradient statistics "
                "require a finite batch"
            )
        return
    if any(value is None for value in statistics):
        raise ValueError(
            "invalid training metrics record: gradient statistics are incomplete"
        )
    mean, maximum, percentile = (
        _number(value, "gradient statistic") for value in statistics
    )
    if mean > maximum or percentile > maximum:
        raise ValueError(
            "invalid training metrics record: gradient statistics differ"
        )


def _target_series(document: JsonObject, field: str) -> list[JsonObject]:
    value = document[field]
    if not isinstance(value, list):
        raise ValueError(f"metrics field {field} must be an array")
    result: list[JsonObject] = []
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, dict):
            raise ValueError(f"metrics field {field}[{index}] must be an object")
        typed_item = cast(JsonObject, item)
        target = typed_item.get("target")
        if not isinstance(target, dict):
            raise ValueError(
                f"metrics field {field}[{index}].target must be an object"
            )
        typed_target = cast(JsonObject, target)
        if typed_target.get("index") != index:
            raise ValueError(
                f"metrics field {field}[{index}] target identity differs"
            )
        result.append(typed_item)
    return result


def validate_point_document(document: object) -> JsonObject:
    return _validate_document(document, _POINT_VALIDATOR, "metrics point")


def project_training_points(
    record: JsonObject,
    *,
    deployment_id: str,
) -> tuple[JsonObject, ...]:
    record = validate_training_record(record)
    points: list[JsonObject] = []
    for metric in _METRICS:
        value = record[metric.field]
        if value is None:
            continue
        points.append(_metric_point(
            record,
            deployment_id=deployment_id,
            name=metric.name,
            value=cast(JsonValue, value),
            unit=metric.unit,
        ))
    for item in _target_series(record, "directLosses"):
        points.append(_metric_point(
            record,
            deployment_id=deployment_id,
            name="training.loss.direct",
            value=item["value"],
            unit="1",
            target=cast(JsonObject, item["target"]),
        ))
    auxiliary_losses = record["auxiliaryLosses"]
    if not isinstance(auxiliary_losses, list):
        raise ValueError("metrics field auxiliaryLosses must be an array")
    for item in cast(list[object], auxiliary_losses):
        if not isinstance(item, dict):
            raise ValueError("auxiliary loss metric must be an object")
        auxiliary = cast(JsonObject, item)
        operator = _string(auxiliary, "operator")
        points.append(_metric_point(
            record,
            deployment_id=deployment_id,
            name="training.loss.auxiliary",
            value=auxiliary["value"],
            unit="1",
            operator=operator,
        ))
    for item in _target_series(record, "targetMetrics"):
        target = cast(JsonObject, item["target"])
        for field, name in (
            ("mae", "training.target.mae"),
            ("rmse", "training.target.rmse"),
        ):
            points.append(_metric_point(
                record,
                deployment_id=deployment_id,
                name=name,
                value=item[field],
                unit="1",
                target=target,
            ))
    interactions = record["gradientInteractions"]
    if interactions is not None:
        if not isinstance(interactions, dict):
            raise ValueError("gradientInteractions must be an object or null")
        gradient = cast(JsonObject, interactions)
        for component in _object_list(gradient, "components"):
            points.append(_metric_point(
                record,
                deployment_id=deployment_id,
                name="training.gradient.component.norm",
                value=component["meanNorm"],
                unit="1",
                gradient_component=_string(component, "name"),
            ))
        for pair in _object_list(gradient, "pairs"):
            pair_identity: JsonObject = {
                "left": _string(pair, "left"),
                "right": _string(pair, "right"),
            }
            points.append(_metric_point(
                record,
                deployment_id=deployment_id,
                name="training.gradient.pair.cosine",
                value=pair["meanCosine"],
                unit="1",
                gradient_pair=pair_identity,
            ))
    return tuple(points)


def _metric_point(
    record: JsonObject,
    *,
    deployment_id: str,
    name: str,
    value: JsonValue,
    unit: str,
    target: JsonObject | None = None,
    operator: str | None = None,
    gradient_component: str | None = None,
    gradient_pair: JsonObject | None = None,
) -> JsonObject:
    target_index = None if target is None else _integer(target, "index")
    identity: list[JsonValue] = [
        POINT_DOCUMENT_SCHEMA,
        "transformer",
        deployment_id,
        _string(record, "jobId"),
        _string(record, "attemptId"),
        None if record["frame"] is None else str(_integer(record, "frame")),
        str(_integer(record, "epoch")),
        str(_integer(record, "step")),
        name,
        (
            None
            if target_index is None
            else "target:" + str(target_index)
        ),
        None if operator is None else "operator:" + operator,
        (
            None
            if gradient_component is None
            else "component:" + gradient_component
        ),
        (
            None
            if gradient_pair is None
            else "pair:"
            + _string(gradient_pair, "left")
            + ":"
            + _string(gradient_pair, "right")
        ),
    ]
    point: JsonObject = {
        "@timestamp": _string(record, "recordedAt"),
        "schema": POINT_DOCUMENT_SCHEMA,
        "eventId": _jcs_sha256(identity),
        "deploymentId": deployment_id,
        "runId": _string(record, "jobId"),
        "source": "transformer",
        "eventKind": "training.epoch",
        "transformerJobId": _string(record, "jobId"),
        "attemptId": _string(record, "attemptId"),
        "attempt": _integer(record, "attempt"),
        "modelRef": _string(record, "modelRef"),
        "epoch": _integer(record, "epoch"),
        "step": _integer(record, "step"),
        "targets": list(_string_list(record, "targets")),
        "dataContractSha256": _string(record, "dataContractSha256"),
        "objectiveConfigSha256": _string(record, "objectiveConfigSha256"),
        "checkpointFormat": _string(record, "checkpointFormat"),
        "transformerVersion": _string(record, "transformerVersion"),
        "transformerGitCommit": _string(record, "transformerGitCommit"),
        "metric": {"name": name, "value": value, "unit": unit},
        "attributes": {
            "checkpointBest": _boolean(record, "checkpoint_best"),
            "shouldStop": _boolean(record, "should_stop"),
            "bestSelectionScore": record["best_selection_score"],
        },
    }
    if record["frame"] is not None:
        point["frame"] = _integer(record, "frame")
    if target is not None:
        point["target"] = dict(target)
    if operator is not None:
        point["operator"] = operator
    if gradient_component is not None:
        point["gradientComponent"] = gradient_component
    if gradient_pair is not None:
        point["gradientPair"] = dict(gradient_pair)
    point["documentSha256"] = _document_sha256(point)
    return validate_point_document(point)


def _validate_document(
    document: object,
    validator: Draft202012Validator,
    name: str,
) -> JsonObject:
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be an object")
    mapping = cast(dict[object, object], document)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    typed_document = cast(JsonObject, document)
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
        raise ValueError(f"invalid {name}: {prefix}{error.message}")
    _validate_finite_numbers(typed_document, name=name)
    return typed_document


def _validate_finite_numbers(value: JsonValue, *, name: str) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _validate_finite_numbers(nested, name=name)
    elif isinstance(value, list):
        for nested in value:
            _validate_finite_numbers(nested, name=name)
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(value)
    ):
        raise ValueError(f"invalid {name}: numeric values must be finite")


def _document_sha256(document: JsonObject) -> str:
    return _jcs_sha256(document)


def _jcs_sha256(value: JsonValue) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _utc_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _string(document: JsonObject, field: str) -> str:
    value = document[field]
    if not isinstance(value, str):
        raise ValueError(f"metrics field {field} must be a string")
    return value


def _integer(document: JsonObject, field: str) -> int:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metrics field {field} must be an integer")
    return value


def _string_list(document: JsonObject, field: str) -> tuple[str, ...]:
    value = document[field]
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"metrics field {field} must contain strings")
    return cast(tuple[str, ...], tuple(value))


def _target_name(item: JsonObject) -> str:
    target = item["target"]
    if not isinstance(target, dict):
        raise ValueError("metrics target must be an object")
    return _string(cast(JsonObject, target), "name")


def _object_list(document: JsonObject, field: str) -> tuple[JsonObject, ...]:
    value = document[field]
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"metrics field {field} must contain objects")
    return tuple(cast(JsonObject, item) for item in value)


def _boolean(document: JsonObject, field: str) -> bool:
    value = document[field]
    if not isinstance(value, bool):
        raise ValueError(f"metrics field {field} must be a boolean")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics field {field} must be a number")
    return float(value)
