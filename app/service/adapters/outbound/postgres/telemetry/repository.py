from __future__ import annotations

import math
from datetime import datetime

from sqlalchemy import func, select

from app.contracts.json_types import JsonObject
from app.contracts.metrics.fit_run.v9 import PROJECTION_VERSION
from app.service.adapters.outbound.postgres.ledger.support import (
    advisory_lock,
    canonical_uuid,
    digest,
    json_value,
    now as timestamp_now,
    positive,
    validate_relative_path,
)
from app.service.adapters.outbound.postgres.models import (
    FitRunSummaryArtifact,
    Job,
    JobAttempt,
    JobInput,
    MetricsOutboxEntry,
    PublishedModel,
    TrainingMetricInterval,
    TrainingMetricsArtifact,
    TrainingRecoveryCheckpoint,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.application.telemetry.records import (
    FitRunSummarySource,
    TrainingMetricIntervalRecord,
)
from app.service.domain.errors import conflict, failed_precondition
from app.service.domain.job import ExecutionState, InputState


class PostgresTrainingTelemetry:
    """Own optional training telemetry persistence and outbox admission."""

    def __init__(self, database: Database) -> None:
        self.database = database

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
        now: float | None = None,
    ) -> bool:
        positive(attempt, "attempt")
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        positive(generation, "generation")
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        for value, label in (
            (checkpoint_serialization_ms, "checkpoint_serialization_ms"),
            (checkpoint_publication_ms, "checkpoint_publication_ms"),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        metrics_value = json_value(metrics)
        if (
            metrics_value.get("epoch") != generation
            or metrics_value.get("step") != global_step
        ):
            raise ValueError("training metrics identity differs from recovery progress")
        recorded_at = timestamp_now(now)
        with self.database.transaction() as session:
            checkpoint = session.get(
                TrainingRecoveryCheckpoint,
                (job_id, generation),
            )
            if (
                checkpoint is None
                or checkpoint.attempt != attempt
                or checkpoint.global_step != global_step
            ):
                raise failed_precondition(
                    "training metrics require the matching recovery checkpoint"
                )
            attempt_record = session.get(JobAttempt, (job_id, attempt))
            if attempt_record is None or attempt_record.attempt_id != attempt_id:
                raise failed_precondition("training metrics attempt identity differs")
            existing = session.get(
                TrainingMetricInterval,
                (job_id, generation),
            )
            if existing is not None:
                if _same_interval(
                    existing,
                    attempt=attempt,
                    attempt_id=attempt_id,
                    metrics=metrics_value,
                    checkpoint_serialization_ms=checkpoint_serialization_ms,
                ):
                    return True
                raise conflict("training metric interval already exists")
            session.add(
                TrainingMetricInterval(
                    job_id=job_id,
                    generation=generation,
                    attempt=attempt,
                    attempt_id=attempt_id,
                    metrics=metrics_value,
                    checkpoint_serialization_ms=checkpoint_serialization_ms,
                    checkpoint_publication_ms=checkpoint_publication_ms,
                    recorded_at=recorded_at,
                )
            )
            session.flush()
            return False

    def epoch_intervals(
        self,
        job_id: str,
    ) -> list[TrainingMetricIntervalRecord]:
        with self.database.session() as session:
            rows = session.scalars(
                select(TrainingMetricInterval)
                .where(TrainingMetricInterval.job_id == job_id)
                .order_by(TrainingMetricInterval.generation)
            )
            return [_interval_record(row) for row in rows]

    def fit_run_summary(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        now: float | None = None,
    ) -> FitRunSummarySource:
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        with self.database.session() as session:
            job = session.get(Job, job_id)
            if (
                job is None
                or job.operation != "fit"
                or job.execution_state
                not in (
                    ExecutionState.RUNNING.value,
                    ExecutionState.SUCCEEDED.value,
                )
                or job.input_state != InputState.CLOSED.value
                or job.attempt != attempt
                or job.input_closed_at is None
            ):
                raise failed_precondition(
                    "fit run summary requires the completed closed attempt"
                )
            publication_boundary = (
                timestamp_now(now)
                if job.execution_state == ExecutionState.RUNNING.value
                else job.finished_at
            )
            if publication_boundary is None:
                raise failed_precondition("fit publication boundary is unavailable")
            attempts = session.scalars(
                select(JobAttempt)
                .where(JobAttempt.job_id == job_id)
                .order_by(JobAttempt.attempt)
            ).all()
            current = attempts[-1] if attempts else None
            if (
                current is None
                or current.attempt != attempt
                or current.attempt_id != attempt_id
                or current.worker_completed_at is None
            ):
                raise failed_precondition(
                    "fit worker completion boundary is unavailable"
                )
            first_input = session.scalar(
                select(func.min(JobInput.committed_at)).where(JobInput.job_id == job_id)
            )
            if first_input is None:
                raise failed_precondition("fit input boundary is unavailable")
            intervals = session.scalars(
                select(TrainingMetricInterval)
                .where(TrainingMetricInterval.job_id == job_id)
                .order_by(TrainingMetricInterval.generation)
            ).all()
            if not intervals:
                raise failed_precondition("fit training metrics are unavailable")

            queue_wait_ms = 0.0
            worker_startup_ms = 0.0
            for item in attempts:
                startup_end = item.worker_ready_at or item.finished_at
                if startup_end is None:
                    raise failed_precondition("attempt startup boundary is unavailable")
                queue_wait_ms += _duration_ms(
                    item.queue_entered_at,
                    item.claimed_at,
                    "queue wait",
                )
                worker_startup_ms += _duration_ms(
                    item.claimed_at,
                    startup_end,
                    "worker startup",
                )

            training_ms = 0.0
            checkpoint_serialization_ms = 0.0
            checkpoint_publication_ms = 0.0
            for interval in intervals:
                training_ms += _finite_nonnegative(
                    interval.metrics.get("elapsedMs"),
                    "training elapsedMs",
                )
                checkpoint_serialization_ms += _finite_nonnegative(
                    interval.checkpoint_serialization_ms,
                    "checkpoint serialization duration",
                )
                checkpoint_publication_ms += _finite_nonnegative(
                    interval.checkpoint_publication_ms,
                    "checkpoint publication duration",
                )

            return FitRunSummarySource(
                job_id=job_id,
                attempt_id=attempt_id,
                attempt=attempt,
                created_at=job.created_at.timestamp(),
                first_input_committed_at=first_input.timestamp(),
                input_closed_at=job.input_closed_at.timestamp(),
                worker_completed_at=current.worker_completed_at.timestamp(),
                publication_boundary_at=publication_boundary.timestamp(),
                queue_wait_ms=queue_wait_ms,
                worker_startup_ms=worker_startup_ms,
                training_ms=training_ms,
                checkpoint_serialization_ms=checkpoint_serialization_ms,
                checkpoint_publication_ms=checkpoint_publication_ms,
                attempt_count=len(attempts),
                recovery_count=sum(
                    item.resume_generation is not None for item in attempts
                ),
                input_payload_count=job.payload_count,
                input_chunks=job.total_chunks,
                input_rows=job.total_rows,
                native_rows=tuple(job.total_native_rows),
                input_bytes=job.total_bytes,
            )

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
    ) -> bool:
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        validate_relative_path(metrics_path)
        validate_relative_path(run_summary_path)
        for value, label in (
            (attempt, "attempt"),
            (metrics_byte_count, "metrics_byte_count"),
            (metrics_row_count, "metrics_row_count"),
            (run_summary_byte_count, "run_summary_byte_count"),
            (max_outbox_entries, "max_outbox_entries"),
            (max_outbox_bytes, "max_outbox_bytes"),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        digest(metrics_sha256, "metrics_sha256")
        digest(run_summary_sha256, "run_summary_sha256")
        if not all(
            (
                metrics_format,
                metrics_media_type,
                run_summary_format,
                run_summary_media_type,
                application_version,
            )
        ):
            raise ValueError("metrics artifact metadata must not be empty")
        if len(git_commit) != 40 or any(
            character not in "0123456789abcdef" for character in git_commit
        ):
            raise ValueError("git_commit must be a lowercase 40-character digest")
        created_at = timestamp_now(now)
        with self.database.transaction() as session:
            model = session.get(PublishedModel, model_ref, with_for_update=True)
            if model is None or model.producing_job_id != job_id:
                return False
            advisory_lock(session, "metrics-outbox-admission")
            queued_entries, queued_bytes = session.execute(
                select(
                    func.count(MetricsOutboxEntry.job_id),
                    func.coalesce(
                        func.sum(
                            TrainingMetricsArtifact.bytes + FitRunSummaryArtifact.bytes
                        ),
                        0,
                    ),
                )
                .join(
                    TrainingMetricsArtifact,
                    TrainingMetricsArtifact.job_id == MetricsOutboxEntry.job_id,
                )
                .join(
                    FitRunSummaryArtifact,
                    FitRunSummaryArtifact.job_id == MetricsOutboxEntry.job_id,
                )
            ).one()
            new_bytes = metrics_byte_count + run_summary_byte_count
            if (
                int(queued_entries) >= max_outbox_entries
                or int(queued_bytes) + new_bytes > max_outbox_bytes
            ):
                return False
            session.add(
                TrainingMetricsArtifact(
                    job_id=job_id,
                    model_ref=model_ref,
                    format=metrics_format,
                    media_type=metrics_media_type,
                    relative_path=metrics_path,
                    bytes=metrics_byte_count,
                    sha256=metrics_sha256,
                    row_count=metrics_row_count,
                    attempt_id=attempt_id,
                    attempt=attempt,
                    application_version=application_version,
                    git_commit=git_commit,
                    created_at=created_at,
                )
            )
            session.add(
                FitRunSummaryArtifact(
                    job_id=job_id,
                    model_ref=model_ref,
                    format=run_summary_format,
                    media_type=run_summary_media_type,
                    relative_path=run_summary_path,
                    bytes=run_summary_byte_count,
                    sha256=run_summary_sha256,
                    attempt_id=attempt_id,
                    attempt=attempt,
                    application_version=application_version,
                    git_commit=git_commit,
                    created_at=created_at,
                )
            )
            session.flush()
            session.add(
                MetricsOutboxEntry(
                    job_id=job_id,
                    projection_version=PROJECTION_VERSION,
                    status="PENDING",
                    cursor=0,
                    attempts=0,
                    next_attempt_at=created_at,
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
            session.flush()
            return True


def _duration_ms(start: datetime, end: datetime, label: str) -> float:
    return _finite_nonnegative(
        (end - start).total_seconds() * 1000.0,
        label,
    )


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise failed_precondition(f"{label} is unavailable or invalid")
    return float(value)


def _interval_record(
    value: TrainingMetricInterval,
) -> TrainingMetricIntervalRecord:
    if (
        value.checkpoint_serialization_ms is None
        or value.checkpoint_publication_ms is None
    ):
        raise ValueError("training metric interval predates checkpoint timing contract")
    return TrainingMetricIntervalRecord(
        job_id=value.job_id,
        generation=value.generation,
        attempt=value.attempt,
        attempt_id=value.attempt_id,
        metrics=dict(value.metrics),
        recorded_at=value.recorded_at.timestamp(),
        checkpoint_serialization_ms=value.checkpoint_serialization_ms,
        checkpoint_publication_ms=value.checkpoint_publication_ms,
    )


def _same_interval(
    value: TrainingMetricInterval,
    *,
    attempt: int,
    attempt_id: str,
    metrics: JsonObject,
    checkpoint_serialization_ms: float,
) -> bool:
    return (
        value.attempt == attempt
        and value.attempt_id == attempt_id
        and value.metrics == metrics
        and value.checkpoint_serialization_ms == checkpoint_serialization_ms
    )


__all__ = ["PostgresTrainingTelemetry"]
