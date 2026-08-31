from __future__ import annotations

import math
import os
from contextlib import AbstractContextManager
from typing import BinaryIO, Protocol

from app.contracts.json_types import JsonObject
from app.contracts.worker.v8.objective import (
    CHECKPOINT_FORMAT,
    objective_from_ml_contract,
)
from app.service.adapters.outbound.artifacts.telemetry.run_summary import (
    publish_fit_run_summary,
)
from app.service.adapters.outbound.artifacts.telemetry.training_metrics import (
    publish_training_metrics,
)
from app.service.application.ports.artifacts import PublishedModelArtifacts
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.telemetry import TrainingTelemetryRepository
from app.service.domain.records import ExecutionJobRecord

_OUTBOX_ENTRY_LIMIT = 10_000
_OUTBOX_BYTE_LIMIT = 10 * 1024 * 1024 * 1024


class _TelemetrySpool(Protocol):
    def staged_file(
        self,
        destination: str,
    ) -> AbstractContextManager[tuple[BinaryIO, str]]: ...

    def atomic_write_json(
        self,
        destination: str,
        document: JsonObject,
    ) -> str: ...

    def telemetry_metrics_path(self, job_id: str) -> str: ...

    def telemetry_run_summary_path(self, job_id: str) -> str: ...

    def telemetry_relative_path(self, absolute_path: str) -> str: ...

    def remove(self, path: str) -> bool: ...


class FitRunTelemetryPublisher:
    """Finalize optional run-owned telemetry after core model publication."""

    def __init__(
        self,
        repository: TrainingTelemetryRepository,
        spool: _TelemetrySpool,
        *,
        logger: EventLogger,
        metrics: OperationalMetricSink,
        application_version: str,
        git_commit: str,
    ) -> None:
        self.repository = repository
        self.spool = spool
        self.logger = logger
        self.metrics = metrics
        self.application_version = application_version
        self.git_commit = git_commit

    def publish(
        self,
        job: ExecutionJobRecord,
        worker_result: JsonObject,
        model: PublishedModelArtifacts,
    ) -> None:
        metrics_path = self.spool.telemetry_metrics_path(job.job_id)
        run_summary_path = self.spool.telemetry_run_summary_path(job.job_id)
        try:
            targets = objective_from_ml_contract(model.ml_contract).targets
            checkpoint_serialization_ms = _nonnegative_number(
                worker_result.get("checkpointSerializationMs"),
                "fit checkpoint serialization duration",
            )
            summary_source = self.repository.fit_run_summary(
                job.job_id,
                job.attempt,
                attempt_id=_attempt_id(job),
            )
            training_metrics = publish_training_metrics(
                self.spool,
                metrics_path,
                self.repository.epoch_intervals(job.job_id),
                job_id=job.job_id,
                model_ref=model.model_ref,
                data_contract_sha256=_string(
                    job.data_contract.get("data_contract_sha256"),
                    "fit data contract sha256",
                ),
                objective_config_sha256=_string(
                    model.ml_contract.get("objectiveConfigSha256"),
                    "fit objective config sha256",
                ),
                checkpoint_format=CHECKPOINT_FORMAT,
                application_version=self.application_version,
                git_commit=self.git_commit,
                targets=targets,
            )
            run_summary = publish_fit_run_summary(
                self.spool,
                run_summary_path,
                summary_source,
                model_ref=model.model_ref,
                data_contract_sha256=_string(
                    job.data_contract.get("data_contract_sha256"),
                    "fit data contract sha256",
                ),
                objective_config_sha256=_string(
                    model.ml_contract.get("objectiveConfigSha256"),
                    "fit objective config sha256",
                ),
                checkpoint_format=CHECKPOINT_FORMAT,
                application_version=self.application_version,
                git_commit=self.git_commit,
                targets=targets,
                terminal_checkpoint_serialization_ms=(
                    checkpoint_serialization_ms
                ),
                terminal_checkpoint_publication_ms=(
                    model.checkpoint_publication_ms
                ),
            )
            registered = self.repository.register_run_artifacts(
                model_ref=model.model_ref,
                job_id=job.job_id,
                attempt_id=_attempt_id(job),
                attempt=job.attempt,
                metrics_path=self.spool.telemetry_relative_path(metrics_path),
                metrics_format=training_metrics.format,
                metrics_media_type=training_metrics.media_type,
                metrics_byte_count=training_metrics.byte_count,
                metrics_sha256=training_metrics.sha256,
                metrics_row_count=training_metrics.row_count,
                run_summary_path=self.spool.telemetry_relative_path(
                    run_summary_path
                ),
                run_summary_format=run_summary.format,
                run_summary_media_type=run_summary.media_type,
                run_summary_byte_count=run_summary.byte_count,
                run_summary_sha256=run_summary.sha256,
                application_version=self.application_version,
                git_commit=self.git_commit,
                max_outbox_entries=_OUTBOX_ENTRY_LIMIT,
                max_outbox_bytes=_OUTBOX_BYTE_LIMIT,
                now=summary_source.publication_boundary_at,
            )
        except Exception as exc:
            self._discard_files(metrics_path, run_summary_path)
            self.metrics.add("trainingTelemetryCollectionErrors")
            self.logger.event(
                "metrics.collection.failed",
                jobId=job.job_id,
                phase="model-artifact",
                errorType=type(exc).__name__,
            )
            return
        if registered:
            self.metrics.add(
                "trainingMetricsArtifactBytes",
                training_metrics.byte_count,
            )
            self.metrics.add(
                "fitRunSummaryArtifactBytes",
                run_summary.byte_count,
            )
            return
        self._discard_files(metrics_path, run_summary_path)
        self.metrics.add("trainingTelemetryDropped")
        self.logger.event(
            "metrics.outbox.dropped",
            jobId=job.job_id,
            modelRef=model.model_ref,
        )

    def _discard_files(self, *paths: str) -> None:
        for path in paths:
            try:
                self.spool.remove(path)
            except OSError as exc:
                self.metrics.add("trainingTelemetryCleanupErrors")
                self.logger.event(
                    "metrics.cleanup.failed",
                    artifact=os.path.basename(path),
                    errorType=type(exc).__name__,
                )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _attempt_id(job: ExecutionJobRecord) -> str:
    if job.attempt_id is None:
        raise ValueError("worker attempt identity is unavailable")
    return job.attempt_id


__all__ = ["FitRunTelemetryPublisher"]
