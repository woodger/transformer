from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol

from app.contracts.json_types import JsonObject
from app.contracts.metrics.fit_run.v1 import (
    PROJECTION_VERSION as LEGACY_RUN_PROJECTION_VERSION,
    RUN_INDEX as LEGACY_RUN_INDEX,
    SUMMARY_FORMAT as LEGACY_SUMMARY_FORMAT,
    SUMMARY_MEDIA_TYPE as LEGACY_SUMMARY_MEDIA_TYPE,
    build_run_document as build_legacy_run_document,
    validate_run_summary as validate_legacy_run_summary,
)
from app.contracts.metrics.fit_run.v2 import (
    PROJECTION_VERSION as CURRENT_PROJECTION_VERSION,
    RUN_INDEX,
    SUMMARY_FORMAT,
    SUMMARY_MEDIA_TYPE,
    build_run_document,
    validate_run_summary,
)
from app.contracts.metrics.v1 import (
    ARTIFACT_FORMAT as LEGACY_ARTIFACT_FORMAT,
    ARTIFACT_INDEX as LEGACY_ARTIFACT_INDEX,
    ARTIFACT_MEDIA_TYPE as LEGACY_ARTIFACT_MEDIA_TYPE,
    POINT_INDEX as LEGACY_POINT_INDEX,
    PROJECTION_VERSION as LEGACY_EPOCH_PROJECTION_VERSION,
    build_artifact_document as build_legacy_artifact_document,
    project_training_points as project_legacy_training_points,
    validate_training_record as validate_legacy_training_record,
)
from app.contracts.metrics.v2 import (
    ARTIFACT_FORMAT,
    ARTIFACT_INDEX,
    ARTIFACT_MEDIA_TYPE,
    POINT_INDEX,
    build_artifact_document,
    project_training_points,
    validate_training_record,
)
from app.service.domain.records import MetricsOutboxRecord

_COPY_CHUNK_BYTES = 1024 * 1024


class _ModelStorage(Protocol):
    def model_absolute_path(self, relative_path: object) -> str: ...


class ModelMetricsProjection:
    """Project one verified immutable model artifact into OpenSearch documents."""

    def __init__(self, storage: _ModelStorage) -> None:
        self.storage = storage

    def points(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> tuple[JsonObject, ...]:
        rows = self._rows(entry)
        if entry.projection_version == CURRENT_PROJECTION_VERSION:
            return tuple(
                point
                for row in rows
                for point in project_training_points(
                    row,
                    deployment_id=deployment_id,
                )
            )
        return tuple(
            point
            for row in rows
            for point in project_legacy_training_points(
                row,
                deployment_id=deployment_id,
            )
        )

    def artifact_document(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> JsonObject:
        artifact = entry.artifact
        builder = (
            build_artifact_document
            if entry.projection_version == CURRENT_PROJECTION_VERSION
            else build_legacy_artifact_document
        )
        return builder(
            deployment_id=deployment_id,
            model_ref=artifact.model_ref,
            job_id=artifact.job_id,
            attempt_id=artifact.attempt_id,
            attempt=artifact.attempt,
            application_version=artifact.application_version,
            git_commit=artifact.git_commit,
            byte_count=artifact.byte_count,
            sha256=artifact.sha256,
            row_count=artifact.row_count,
            created_at=artifact.created_at,
        )

    def run_summary_document(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> JsonObject | None:
        if entry.projection_version == LEGACY_EPOCH_PROJECTION_VERSION:
            if entry.run_summary is not None:
                raise ValueError(
                    "legacy metrics projection contains a run summary"
                )
            return None
        if entry.projection_version not in (
            LEGACY_RUN_PROJECTION_VERSION,
            CURRENT_PROJECTION_VERSION,
        ):
            raise ValueError("unsupported metrics projection version")
        current = entry.projection_version == CURRENT_PROJECTION_VERSION
        summary_format = SUMMARY_FORMAT if current else LEGACY_SUMMARY_FORMAT
        summary_media_type = (
            SUMMARY_MEDIA_TYPE if current else LEGACY_SUMMARY_MEDIA_TYPE
        )
        artifact = entry.run_summary
        if artifact is None:
            raise ValueError("fit run summary artifact is unavailable")
        if (
            artifact.format != summary_format
            or artifact.media_type != summary_media_type
            or artifact.model_ref != entry.artifact.model_ref
            or artifact.job_id != entry.artifact.job_id
            or artifact.attempt_id != entry.artifact.attempt_id
            or artifact.attempt != entry.artifact.attempt
            or artifact.application_version
            != entry.artifact.application_version
            or artifact.git_commit != entry.artifact.git_commit
        ):
            raise ValueError("fit run summary metadata is inconsistent")
        path = self.storage.model_absolute_path(artifact.relative_path)
        if (
            os.path.getsize(path) != artifact.byte_count
            or _sha256_file(path) != artifact.sha256
        ):
            raise ValueError("fit run summary integrity check failed")
        try:
            with open(path, encoding="utf-8") as source:
                loaded: object = json.load(source)
                summary = (
                    validate_run_summary(loaded)
                    if current
                    else validate_legacy_run_summary(loaded)
                )
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
        if current:
            return build_run_document(
                summary,
                deployment_id=deployment_id,
                byte_count=artifact.byte_count,
                sha256=artifact.sha256,
            )
        return build_legacy_run_document(
            summary,
            deployment_id=deployment_id,
            byte_count=artifact.byte_count,
            sha256=artifact.sha256,
        )

    def point_index(self, entry: MetricsOutboxRecord) -> str:
        return (
            POINT_INDEX
            if entry.projection_version == CURRENT_PROJECTION_VERSION
            else LEGACY_POINT_INDEX
        )

    def artifact_index(self, entry: MetricsOutboxRecord) -> str:
        return (
            ARTIFACT_INDEX
            if entry.projection_version == CURRENT_PROJECTION_VERSION
            else LEGACY_ARTIFACT_INDEX
        )

    def run_summary_index(self, entry: MetricsOutboxRecord) -> str:
        return (
            RUN_INDEX
            if entry.projection_version == CURRENT_PROJECTION_VERSION
            else LEGACY_RUN_INDEX
        )

    def _rows(self, entry: MetricsOutboxRecord) -> tuple[JsonObject, ...]:
        artifact = entry.artifact
        if entry.projection_version not in (
            LEGACY_EPOCH_PROJECTION_VERSION,
            LEGACY_RUN_PROJECTION_VERSION,
            CURRENT_PROJECTION_VERSION,
        ):
            raise ValueError("unsupported metrics projection version")
        if (
            entry.projection_version in (
                LEGACY_RUN_PROJECTION_VERSION,
                CURRENT_PROJECTION_VERSION,
            )
            and entry.run_summary is None
        ):
            raise ValueError("fit run summary artifact is unavailable")
        if (
            artifact.format
            != (
                ARTIFACT_FORMAT
                if entry.projection_version == CURRENT_PROJECTION_VERSION
                else LEGACY_ARTIFACT_FORMAT
            )
            or artifact.media_type
            != (
                ARTIFACT_MEDIA_TYPE
                if entry.projection_version == CURRENT_PROJECTION_VERSION
                else LEGACY_ARTIFACT_MEDIA_TYPE
            )
        ):
            raise ValueError("unsupported training metrics artifact format")
        path = self.storage.model_absolute_path(artifact.relative_path)
        if (
            os.path.getsize(path) != artifact.byte_count
            or _sha256_file(path) != artifact.sha256
        ):
            raise ValueError("training metrics artifact integrity check failed")
        rows: list[JsonObject] = []
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.endswith("\n"):
                    raise ValueError(
                        "training metrics artifact line is not terminated"
                    )
                try:
                    loaded: object = json.loads(line)
                    row = (
                        validate_training_record(loaded)
                        if entry.projection_version == CURRENT_PROJECTION_VERSION
                        else validate_legacy_training_record(loaded)
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        "training metrics artifact contains an invalid record"
                    ) from exc
                if (
                    row["jobId"] != artifact.job_id
                    or row["modelRef"] != artifact.model_ref
                    or row["transformerVersion"]
                    != artifact.application_version
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


__all__ = ["ModelMetricsProjection"]
