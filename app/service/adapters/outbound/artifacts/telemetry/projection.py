from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol

from app.contracts.json_types import JsonObject
from app.contracts.metrics.fit_run.v6 import (
    PROJECTION_VERSION,
    SUMMARY_FORMAT,
    SUMMARY_MEDIA_TYPE,
    build_run_document,
    validate_run_summary,
)
from app.contracts.metrics.v6 import (
    ARTIFACT_FORMAT,
    ARTIFACT_MEDIA_TYPE,
    project_training_points,
    validate_training_record,
)
from app.service.application.telemetry.records import MetricsOutboxRecord

_COPY_CHUNK_BYTES = 1024 * 1024


class _TelemetryStorage(Protocol):
    def telemetry_absolute_path(self, relative_path: object) -> str: ...


class TrainingMetricsProjection:
    """Project one verified immutable fit-run artifact into OpenSearch."""

    def __init__(self, storage: _TelemetryStorage) -> None:
        self.storage = storage

    def points(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> tuple[JsonObject, ...]:
        rows = self._rows(entry)
        return tuple(
            point
            for row in rows
            for point in project_training_points(
                row,
                deployment_id=deployment_id,
            )
        )

    def run_summary_document(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> JsonObject:
        if entry.projection_version != PROJECTION_VERSION:
            raise ValueError("unsupported metrics projection version")
        artifact = entry.run_summary
        if artifact is None:
            raise ValueError("fit run summary artifact is unavailable")
        if (
            artifact.format != SUMMARY_FORMAT
            or artifact.media_type != SUMMARY_MEDIA_TYPE
            or artifact.model_ref != entry.training_metrics.model_ref
            or artifact.job_id != entry.training_metrics.job_id
            or artifact.attempt_id != entry.training_metrics.attempt_id
            or artifact.attempt != entry.training_metrics.attempt
            or artifact.application_version
            != entry.training_metrics.application_version
            or artifact.git_commit != entry.training_metrics.git_commit
        ):
            raise ValueError("fit run summary metadata is inconsistent")
        path = self.storage.telemetry_absolute_path(artifact.relative_path)
        if (
            os.path.getsize(path) != artifact.byte_count
            or _sha256_file(path) != artifact.sha256
        ):
            raise ValueError("fit run summary integrity check failed")
        try:
            with open(path, encoding="utf-8") as source:
                loaded: object = json.load(source)
                summary = validate_run_summary(loaded)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise ValueError("fit run summary artifact is invalid") from exc
        if (
            summary["jobId"] != artifact.job_id
            or summary["attemptId"] != artifact.attempt_id
            or summary["attempt"] != artifact.attempt
            or summary["modelRef"] != artifact.model_ref
            or summary["transformerVersion"] != artifact.application_version
            or summary["transformerGitCommit"] != artifact.git_commit
        ):
            raise ValueError("fit run summary identity differs from metadata")
        return build_run_document(
            summary,
            deployment_id=deployment_id,
            byte_count=artifact.byte_count,
            sha256=artifact.sha256,
        )

    def _rows(self, entry: MetricsOutboxRecord) -> tuple[JsonObject, ...]:
        artifact = entry.training_metrics
        if entry.projection_version != PROJECTION_VERSION:
            raise ValueError("unsupported metrics projection version")
        if entry.run_summary is None:
            raise ValueError("fit run summary artifact is unavailable")
        if (
            artifact.format != ARTIFACT_FORMAT
            or artifact.media_type != ARTIFACT_MEDIA_TYPE
        ):
            raise ValueError("unsupported training metrics artifact format")
        path = self.storage.telemetry_absolute_path(artifact.relative_path)
        if (
            os.path.getsize(path) != artifact.byte_count
            or _sha256_file(path) != artifact.sha256
        ):
            raise ValueError("training metrics artifact integrity check failed")
        rows: list[JsonObject] = []
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.endswith("\n"):
                    raise ValueError("training metrics artifact line is not terminated")
                try:
                    loaded: object = json.loads(line)
                    row = validate_training_record(loaded)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        "training metrics artifact contains an invalid record"
                    ) from exc
                if (
                    row["jobId"] != artifact.job_id
                    or row["modelRef"] != artifact.model_ref
                    or row["transformerVersion"] != artifact.application_version
                    or row["transformerGitCommit"] != artifact.git_commit
                ):
                    raise ValueError(
                        "training metrics record identity differs from metadata"
                    )
                rows.append(row)
        if len(rows) != artifact.row_count:
            raise ValueError("training metrics artifact row count differs")
        return tuple(rows)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["TrainingMetricsProjection"]
