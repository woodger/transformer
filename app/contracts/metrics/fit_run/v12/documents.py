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

SUMMARY_FORMAT = "transformer.fit-run-summary.v12"
SUMMARY_MEDIA_TYPE = "application/json"
PROJECTION_VERSION = "transformer.metrics.v12"
RUN_DOCUMENT_SCHEMA = "transformer.metrics-fit-run.v12"
RUN_INDEX = "metrics-runs-v12"

_ROOT = Path(__file__).parent
_SEMANTIC_SCHEMAS = _ROOT.parents[2] / "semantic" / "v6" / "schemas"
_CHECKPOINT_SCHEMAS = _ROOT.parents[2] / "checkpoint" / "v13" / "schemas"


def build_run_summary(
    *,
    recorded_at: float,
    job_id: str,
    attempt_id: str,
    attempt: int,
    model_ref: str,
    semantic_digests: JsonObject,
    job_config_sha256: str,
    checkpoint_format: str,
    application_version: str,
    git_commit: str,
    targets: Sequence[str],
    initialization: JsonObject,
    milestones: JsonObject,
    durations: JsonObject,
    counts: JsonObject,
) -> JsonObject:
    return validate_run_summary(
        {
            "format": SUMMARY_FORMAT,
            "recordedAt": _utc_timestamp(recorded_at),
            "jobId": job_id,
            "attemptId": attempt_id,
            "attempt": attempt,
            "modelRef": model_ref,
            "semanticDigests": dict(semantic_digests),
            "jobConfigSha256": job_config_sha256,
            "checkpointFormat": checkpoint_format,
            "transformerVersion": application_version,
            "transformerGitCommit": git_commit,
            "targets": [
                {"index": index, "identity": identity}
                for index, identity in enumerate(targets)
            ],
            "initialization": dict(initialization),
            "milestones": dict(milestones),
            "durations": dict(durations),
            "counts": dict(counts),
        }
    )


def build_run_document(
    summary: JsonObject,
    *,
    deployment_id: str,
    byte_count: int,
    sha256: str,
) -> JsonObject:
    summary = validate_run_summary(summary)
    document: JsonObject = {
        "schema": RUN_DOCUMENT_SCHEMA,
        "summaryId": _jcs_sha256(
            [
                RUN_DOCUMENT_SCHEMA,
                deployment_id,
                cast(str, summary["jobId"]),
                cast(str, summary["modelRef"]),
            ]
        ),
        "deploymentId": deployment_id,
        "runId": summary["jobId"],
        "modelRef": summary["modelRef"],
        "recordedAt": summary["recordedAt"],
        "semanticDigests": summary["semanticDigests"],
        "jobConfigSha256": summary["jobConfigSha256"],
        "checkpointFormat": summary["checkpointFormat"],
        "targets": summary["targets"],
        "initialization": summary["initialization"],
        "milestones": summary["milestones"],
        "durations": summary["durations"],
        "counts": summary["counts"],
        "artifact": {"byteCount": byte_count, "sha256": sha256},
    }
    return validate_run_document(document)


def validate_run_summary(document: object) -> JsonObject:
    typed = _validate(document, "run-summary", "fit run summary")
    _validate_targets(typed)
    return typed


def validate_run_document(document: object) -> JsonObject:
    typed = _validate(document, "run-document", "fit run document")
    _validate_targets(typed)
    return typed


def _validate_targets(document: JsonObject) -> None:
    targets = document["targets"]
    if not isinstance(targets, list):
        raise ValueError("fit run targets must be an array")
    identities: list[str] = []
    for index, item in enumerate(cast(list[object], targets)):
        if not isinstance(item, Mapping):
            raise ValueError("fit run target must be an object")
        target = cast(Mapping[object, object], item)
        if target.get("index") != index or not isinstance(target.get("identity"), str):
            raise ValueError("fit run target order differs")
        identities.append(cast(str, target["identity"]))
    if len(identities) != len(set(identities)):
        raise ValueError("fit run target identities must be unique")


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
    return Draft202012Validator(
        _load_schema(_ROOT / f"{name}.schema.json"),
        registry=_registry(),
        format_checker=FormatChecker(),
    )


@cache
def _registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for directory in (_ROOT, _SEMANTIC_SCHEMAS, _CHECKPOINT_SCHEMAS):
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


def _jcs_sha256(value: JsonValue) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _utc_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "PROJECTION_VERSION",
    "RUN_INDEX",
    "SUMMARY_FORMAT",
    "SUMMARY_MEDIA_TYPE",
    "build_run_document",
    "build_run_summary",
    "validate_run_document",
    "validate_run_summary",
]
