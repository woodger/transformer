from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker

from app.contracts.json_types import JsonObject, JsonValue

SUMMARY_FORMAT = "transformer.fit-run-summary.v2"
SUMMARY_MEDIA_TYPE = "application/json"
PROJECTION_VERSION = "inventory.metrics.v3"
RUN_DOCUMENT_SCHEMA = "inventory.metrics.fit-run.v2"
RUN_INDEX = "metrics-runs-v2"

_ROOT = Path(__file__).parent
_SUMMARY_SCHEMA = json.loads(
    (_ROOT / "run-summary.schema.json").read_text(encoding="utf-8")
)
_SUMMARY_VALIDATOR = Draft202012Validator(
    _SUMMARY_SCHEMA,
    format_checker=FormatChecker(),
)
_RUN_SCHEMA = json.loads(
    (_ROOT / "run-document.schema.json").read_text(encoding="utf-8")
)
for _field in ("milestones", "durations", "counts", "targetStatistics"):
    _RUN_SCHEMA["properties"][_field] = _SUMMARY_SCHEMA["properties"][_field]
_RUN_SCHEMA["$defs"] = _SUMMARY_SCHEMA["$defs"]
_RUN_VALIDATOR = Draft202012Validator(
    _RUN_SCHEMA,
    format_checker=FormatChecker(),
)


def build_run_summary(
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
    milestones: JsonObject,
    durations: JsonObject,
    counts: JsonObject,
    target_statistics: list[JsonObject],
) -> JsonObject:
    return validate_run_summary({
        "format": SUMMARY_FORMAT,
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
        "milestones": milestones,
        "durations": durations,
        "counts": counts,
        "targetStatistics": target_statistics,
    })


def build_run_document(
    summary: JsonObject,
    *,
    deployment_id: str,
    byte_count: int,
    sha256: str,
) -> JsonObject:
    summary = validate_run_summary(summary)
    job_id = _string(summary, "jobId")
    model_ref = _string(summary, "modelRef")
    summary_id = _jcs_sha256([
        RUN_DOCUMENT_SCHEMA,
        "transformer",
        deployment_id,
        job_id,
        model_ref,
    ])
    document: JsonObject = {
        "@timestamp": _string(summary, "recordedAt"),
        "schema": RUN_DOCUMENT_SCHEMA,
        "summaryId": summary_id,
        "deploymentId": deployment_id,
        "runId": job_id,
        "source": "transformer",
        "eventKind": "fit.run.succeeded",
        "transformerJobId": job_id,
        "attemptId": _string(summary, "attemptId"),
        "attempt": _integer(summary, "attempt"),
        "modelRef": model_ref,
        "dataContractSha256": _string(summary, "dataContractSha256"),
        "objectiveConfigSha256": _string(
            summary,
            "objectiveConfigSha256",
        ),
        "checkpointFormat": _string(summary, "checkpointFormat"),
        "transformerVersion": _string(summary, "transformerVersion"),
        "transformerGitCommit": _string(
            summary,
            "transformerGitCommit",
        ),
        "milestones": cast(JsonObject, summary["milestones"]),
        "durations": cast(JsonObject, summary["durations"]),
        "counts": cast(JsonObject, summary["counts"]),
        "targetStatistics": cast(list[JsonValue], summary["targetStatistics"]),
        "artifact": {
            "format": SUMMARY_FORMAT,
            "mediaType": SUMMARY_MEDIA_TYPE,
            "bytes": byte_count,
            "sha256": sha256,
        },
    }
    document["documentSha256"] = _jcs_sha256(document)
    return validate_run_document(document)


def validate_run_summary(document: object) -> JsonObject:
    return _validate(document, _SUMMARY_VALIDATOR, "fit run summary")


def validate_run_document(document: object) -> JsonObject:
    return _validate(document, _RUN_VALIDATOR, "fit run document")


def _validate(
    document: object,
    validator: Draft202012Validator,
    label: str,
) -> JsonObject:
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be an object")
    mapping = cast(dict[object, object], document)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} keys must be strings")
    typed = cast(JsonObject, document)
    errors = sorted(
        validator.iter_errors(typed),  # pyright: ignore[reportUnknownMemberType]
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{location}: " if location else ""
        raise ValueError(f"invalid {label}: {prefix}{error.message}")
    _finite(typed, label)
    _validate_target_statistics(typed, label)
    return typed


def _validate_target_statistics(document: JsonObject, label: str) -> None:
    counts_value = document.get("counts")
    statistics_value = document.get("targetStatistics")
    if not isinstance(counts_value, dict) or not isinstance(
        statistics_value,
        list,
    ):
        raise ValueError(f"invalid {label}: target statistics are unavailable")
    counts = cast(JsonObject, counts_value)
    input_rows = _integer(counts, "inputRows")
    for value in cast(list[JsonValue], statistics_value):
        if not isinstance(value, dict):
            raise ValueError(f"invalid {label}: target statistic must be an object")
        statistic = cast(JsonObject, value)
        count = _integer(statistic, "count")
        zero_count = _integer(statistic, "zeroCount")
        one_count = _integer(statistic, "oneCount")
        minimum = _number(statistic, "min")
        maximum = _number(statistic, "max")
        mean = _number(statistic, "mean")
        if count != input_rows:
            raise ValueError(
                f"invalid {label}: target count differs from input rows"
            )
        if (
            zero_count > count
            or one_count > count
            or zero_count + one_count > count
        ):
            raise ValueError(f"invalid {label}: target value counts differ")
        if minimum > mean or mean > maximum:
            raise ValueError(f"invalid {label}: target moments are inconsistent")


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
        raise ValueError(f"fit run field {field} must be a string")
    return value


def _integer(document: JsonObject, field: str) -> int:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fit run field {field} must be an integer")
    return value


def _number(document: JsonObject, field: str) -> float:
    value = document[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"fit run field {field} must be a number")
    return float(value)
