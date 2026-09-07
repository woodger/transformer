from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.service.application.ports.artifacts import PublishedModelArtifacts
from app.service.application.telemetry.records import (
    FitRunSummarySource,
    MetricsOutboxRecord,
    TelemetryArtifactCleanup,
    TrainingMetricIntervalRecord,
)
from app.service.domain.json_types import JsonObject
from app.service.domain.records import ExecutionJobRecord


class RetryableMetricsDeliveryError(RuntimeError):
    pass


class BlockedMetricsDeliveryError(RuntimeError):
    pass


class MetricsOutboxRepository(Protocol):
    def next_pending(self) -> MetricsOutboxRecord | None: ...

    def advance(
        self,
        job_id: str,
        *,
        expected_cursor: int,
        cursor: int,
    ) -> bool: ...

    def retry(
        self,
        job_id: str,
        *,
        expected_cursor: int,
        delay_seconds: float,
        error_code: str,
        error_message: str,
    ) -> bool: ...

    def block(
        self,
        job_id: str,
        *,
        expected_cursor: int,
        error_code: str,
        error_message: str,
    ) -> bool: ...

    def discard(
        self,
        job_id: str,
        *,
        expected_cursor: int,
        error_code: str,
        error_message: str,
    ) -> bool: ...

    def complete(
        self,
        job_id: str,
        *,
        expected_cursor: int,
    ) -> bool: ...

    def purge_terminal(
        self,
        *,
        older_than_seconds: float,
    ) -> tuple[TelemetryArtifactCleanup, ...]: ...

    def backlog(self) -> tuple[int, int, float | None]: ...

    def retained_run_ids(self) -> set[str]: ...


class TelemetryArtifactStorage(Protocol):
    def remove_telemetry_artifacts(
        self,
        relative_paths: Sequence[str],
    ) -> None: ...

class MetricsArtifactProjection(Protocol):
    def points(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> tuple[JsonObject, ...]: ...

    def run_summary_document(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> JsonObject: ...


class MetricsDocumentSink(Protocol):
    def create_documents(
        self,
        index: str,
        documents: Sequence[JsonObject],
        *,
        id_field: str,
        refresh: bool = False,
    ) -> None: ...


class TrainingTelemetryRepository(Protocol):
    """Persist optional training observations outside job/model lifecycle."""

    def record_epoch_interval(
        self,
        *,
        job_id: str,
        attempt: int,
        attempt_id: str,
        generation: int,
        global_step: int,
        metrics: JsonObject,
        checkpoint_serialization_ms: float,
        checkpoint_publication_ms: float,
    ) -> bool: ...

    def epoch_intervals(
        self,
        job_id: str,
    ) -> list[TrainingMetricIntervalRecord]: ...

    def fit_run_summary(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
    ) -> FitRunSummarySource: ...

    def register_run_artifacts(
        self,
        *,
        model_ref: str,
        job_id: str,
        attempt_id: str,
        attempt: int,
        metrics_path: str,
        metrics_format: str,
        metrics_media_type: str,
        metrics_byte_count: int,
        metrics_sha256: str,
        metrics_row_count: int,
        run_summary_path: str,
        run_summary_format: str,
        run_summary_media_type: str,
        run_summary_byte_count: int,
        run_summary_sha256: str,
        application_version: str,
        git_commit: str,
        max_outbox_entries: int,
        max_outbox_bytes: int,
        now: float | None = None,
    ) -> bool: ...


class FitTelemetryPublisher(Protocol):
    def publish(
        self,
        job: ExecutionJobRecord,
        worker_result: JsonObject,
        model: PublishedModelArtifacts,
    ) -> None: ...


__all__ = [
    "BlockedMetricsDeliveryError",
    "FitTelemetryPublisher",
    "MetricsArtifactProjection",
    "MetricsDocumentSink",
    "MetricsOutboxRepository",
    "RetryableMetricsDeliveryError",
    "TelemetryArtifactStorage",
    "TrainingTelemetryRepository",
]
