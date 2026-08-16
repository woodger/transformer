from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker

from app.contracts.json_types import JsonObject, JsonValue

ARTIFACT_FORMAT = "transformer.training-metrics.v1"
ARTIFACT_MEDIA_TYPE = "application/x-ndjson"
PROJECTION_VERSION = "inventory.metrics.v1"
POINT_DOCUMENT_SCHEMA = "inventory.metrics.point.v1"
ARTIFACT_DOCUMENT_SCHEMA = "inventory.metrics.artifact.v1"
POINT_STREAM = "metrics-points-v1"
ARTIFACT_STREAM = "metrics-artifacts-v1"

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
_ARTIFACT_VALIDATOR = Draft202012Validator(
    json.loads(Path(__file__).with_name("artifact.schema.json").read_text(
        encoding="utf-8"
    )),
    format_checker=FormatChecker(),
)


@dataclass(frozen=True, slots=True)
class _Metric:
    field: str
    name: str
    unit: str = "1"
    target_index: int | None = None


_TARGETS = (
    "mean_return",
    "sigma_return",
    "prob_tp",
    "prob_sl",
    "volatility_next",
    "hitting_prob_tp",
)
_METRICS = (
    _Metric("loss", "training.loss.total"),
    *(
        _Metric(
            f"loss_l{index}",
            f"training.loss.direct.{semantic}",
            target_index=index,
        )
        for index, semantic in enumerate(_TARGETS)
    ),
    _Metric("loss_nll", "training.loss.auxiliary.nll"),
    _Metric("loss_ev", "training.loss.auxiliary.ev"),
    *(
        _Metric(
            f"{semantic}_mae",
            f"training.target.{semantic}.mae",
            target_index=index,
        )
        for index, semantic in enumerate(_TARGETS)
    ),
    *(
        _Metric(
            f"{semantic}_rmse",
            f"training.target.{semantic}.rmse",
            target_index=index,
        )
        for index, semantic in enumerate(_TARGETS)
    ),
    _Metric("selection_score", "training.selection.score"),
    _Metric("grad_norm", "training.gradient.pre_clip_norm"),
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
) -> JsonObject:
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
    for field, value in typed_document.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(value)
        ):
            raise ValueError(
                f"invalid training metrics record: {field} must be finite"
            )
    return typed_document


def validate_point_document(document: object) -> JsonObject:
    return _validate_document(document, _POINT_VALIDATOR, "metrics point")


def validate_artifact_document(document: object) -> JsonObject:
    return _validate_document(
        document,
        _ARTIFACT_VALIDATOR,
        "metrics artifact",
    )


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
        identity: list[JsonValue] = [
            POINT_DOCUMENT_SCHEMA,
            "transformer",
            deployment_id,
            _string(record, "jobId"),
            _string(record, "attemptId"),
            (
                None
                if record["frame"] is None
                else str(_integer(record, "frame"))
            ),
            str(_integer(record, "epoch")),
            str(_integer(record, "step")),
            metric.name,
            None if metric.target_index is None else str(metric.target_index),
        ]
        event_id = _jcs_sha256(identity)
        attributes: JsonObject = {
            "checkpointBest": _boolean(record, "checkpoint_best"),
            "shouldStop": _boolean(record, "should_stop"),
            "bestSelectionScore": record["best_selection_score"],
        }
        point: JsonObject = {
            "@timestamp": _string(record, "recordedAt"),
            "schema": POINT_DOCUMENT_SCHEMA,
            "eventId": event_id,
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
            "lossStage": _integer(record, "loss_stage"),
            "dataContractSha256": _string(record, "dataContractSha256"),
            "objectiveConfigSha256": _string(
                record,
                "objectiveConfigSha256",
            ),
            "checkpointFormat": _string(record, "checkpointFormat"),
            "transformerVersion": _string(record, "transformerVersion"),
            "transformerGitCommit": _string(
                record,
                "transformerGitCommit",
            ),
            "metric": {
                "name": metric.name,
                "value": cast(JsonValue, value),
                "unit": metric.unit,
            },
            "attributes": attributes,
        }
        if record["frame"] is not None:
            point["frame"] = _integer(record, "frame")
        if metric.target_index is not None:
            point["targetIndex"] = metric.target_index
        point["documentSha256"] = _document_sha256(point)
        points.append(validate_point_document(point))
    return tuple(points)


def build_artifact_document(
    *,
    deployment_id: str,
    model_ref: str,
    job_id: str,
    attempt_id: str,
    attempt: int,
    application_version: str,
    git_commit: str,
    byte_count: int,
    sha256: str,
    row_count: int,
    created_at: float,
) -> JsonObject:
    artifact_id = _jcs_sha256([
        ARTIFACT_DOCUMENT_SCHEMA,
        "transformer",
        deployment_id,
        model_ref,
        "training-metrics",
    ])
    document: JsonObject = {
        "@timestamp": _utc_timestamp(created_at),
        "schema": ARTIFACT_DOCUMENT_SCHEMA,
        "artifactId": artifact_id,
        "deploymentId": deployment_id,
        "kind": "training-metrics",
        "owner": "transformer",
        "runId": job_id,
        "transformerJobId": job_id,
        "attemptId": attempt_id,
        "attempt": attempt,
        "modelRef": model_ref,
        "format": ARTIFACT_FORMAT,
        "mediaType": ARTIFACT_MEDIA_TYPE,
        "bytes": byte_count,
        "rowCount": row_count,
        "sha256": sha256,
        "transformerVersion": application_version,
        "transformerGitCommit": git_commit,
    }
    document["documentSha256"] = _document_sha256(document)
    return validate_artifact_document(document)


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


def _boolean(document: JsonObject, field: str) -> bool:
    value = document[field]
    if not isinstance(value, bool):
        raise ValueError(f"metrics field {field} must be a boolean")
    return value
