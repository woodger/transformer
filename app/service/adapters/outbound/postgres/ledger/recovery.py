from __future__ import annotations

import math
from typing import cast

from sqlalchemy import select

from app.contracts.worker.v9.objective import TRAINING_RECOVERY_FORMAT
from app.service.adapters.outbound.postgres.ledger.support import (
    LedgerSessions,
    RowMapping,
    canonical_uuid,
    decode,
    digest,
    now as timestamp_now,
    positive,
    validate_relative_path,
)
from app.service.adapters.outbound.postgres.models import (
    QUEUE_SEQUENCE,
    Job,
    JobAttempt,
    JobInput,
    TrainingRecoveryCheckpoint,
)
from app.service.domain.errors import (
    conflict,
    failed_precondition,
    not_found,
)
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import TrainingRecoveryCheckpointRecord


class RecoveryLedgerSlice:
    """Atomic visibility and retry operations for training recovery."""

    def __init__(self, sessions: LedgerSessions) -> None:
        self.sessions = sessions
        self.database = sessions.database

    def register_checkpoint(
        self,
        *,
        job_id: str,
        attempt: int,
        attempt_id: str,
        generation: int,
        format: str,
        relative_path: str,
        byte_count: int,
        sha256: str,
        completed_epochs: int,
        global_step: int,
        loss: float,
        training_complete: bool,
        now: float | None = None,
    ) -> tuple[TrainingRecoveryCheckpointRecord, bool]:
        positive(attempt, "attempt")
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        positive(generation, "generation")
        positive(byte_count, "bytes")
        positive(completed_epochs, "completed_epochs")
        raw_global_step = cast(object, global_step)
        if (
            isinstance(raw_global_step, bool)
            or not isinstance(raw_global_step, int)
            or raw_global_step < 0
        ):
            raise ValueError("global_step must be a non-negative integer")
        raw_loss = cast(object, loss)
        if (
            isinstance(raw_loss, bool)
            or not isinstance(raw_loss, (int, float))
            or not math.isfinite(raw_loss)
        ):
            raise ValueError("loss must be a finite number")
        raw_training_complete = cast(object, training_complete)
        if not isinstance(raw_training_complete, bool):
            raise ValueError("training_complete must be a boolean")
        if format != TRAINING_RECOVERY_FORMAT:
            raise ValueError("unsupported training recovery format")
        validate_relative_path(relative_path)
        digest(sha256, "sha256")
        created_at = timestamp_now(now)
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update()
            )
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if (
                job.operation != "fit"
                or job.execution_state != ExecutionState.RUNNING.value
                or job.input_state != InputState.CLOSED.value
                or job.attempt != attempt
            ):
                raise failed_precondition(
                    "job is not the active running fit attempt"
                )
            attempt_record = session.get(
                JobAttempt,
                (job_id, attempt),
                with_for_update=True,
            )
            if (
                attempt_record is None
                or attempt_record.status != ExecutionState.RUNNING.value
                or attempt_record.attempt_id != attempt_id
            ):
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            existing = session.get(
                TrainingRecoveryCheckpoint,
                (job_id, generation),
            )
            if existing is not None:
                if _same_checkpoint(
                    existing,
                    attempt=attempt,
                    format=format,
                    relative_path=relative_path,
                    byte_count=byte_count,
                    sha256=sha256,
                    completed_epochs=completed_epochs,
                    global_step=global_step,
                    training_complete=training_complete,
                ):
                    return _record(existing), True
                raise conflict(
                    "training recovery generation already exists"
                )
            latest = session.scalar(
                select(TrainingRecoveryCheckpoint)
                .where(
                    TrainingRecoveryCheckpoint.job_id == job_id
                )
                .order_by(
                    TrainingRecoveryCheckpoint.generation.desc()
                )
                .limit(1)
            )
            if latest is not None and (
                generation <= latest.generation
                or completed_epochs <= latest.completed_epochs
                or global_step < latest.global_step
            ):
                raise failed_precondition(
                    "training recovery progress is not monotonic"
                )
            record = TrainingRecoveryCheckpoint(
                job_id=job_id,
                generation=generation,
                attempt=attempt,
                format=format,
                relative_path=relative_path,
                bytes=byte_count,
                sha256=sha256,
                completed_epochs=completed_epochs,
                global_step=global_step,
                training_complete=training_complete,
                created_at=created_at,
            )
            session.add(record)
            job.revision += 1
            job.progress = {
                "epoch": completed_epochs,
                "step": global_step,
                "loss": float(loss),
            }
            job.updated_at = created_at
            session.flush()
            return _record(record), False

    def latest_checkpoint(
        self,
        job_id: str,
    ) -> TrainingRecoveryCheckpointRecord | None:
        with self.database.session() as session:
            row = session.scalar(
                select(TrainingRecoveryCheckpoint)
                .where(
                    TrainingRecoveryCheckpoint.job_id == job_id
                )
                .order_by(
                    TrainingRecoveryCheckpoint.generation.desc()
                )
                .limit(1)
            )
            return None if row is None else _record(row)

    def list_checkpoints(
        self,
        job_id: str,
    ) -> list[TrainingRecoveryCheckpointRecord]:
        with self.database.session() as session:
            rows = session.scalars(
                select(TrainingRecoveryCheckpoint)
                .where(
                    TrainingRecoveryCheckpoint.job_id == job_id
                )
                .order_by(
                    TrainingRecoveryCheckpoint.generation.desc()
                )
            )
            return [_record(row) for row in rows]

    def referenced_paths(self) -> set[str]:
        with self.database.session() as session:
            paths = set(session.scalars(
                select(TrainingRecoveryCheckpoint.relative_path)
                .join(
                    Job,
                    Job.job_id == TrainingRecoveryCheckpoint.job_id,
                )
                .where(Job.execution_state.not_in((
                    ExecutionState.SUCCEEDED.value,
                    ExecutionState.FAILED.value,
                    ExecutionState.CANCELLED.value,
                )))
            ))
            paths.update(session.scalars(
                select(JobInput.relative_path)
                .join(Job, Job.job_id == JobInput.job_id)
                .where(
                    JobInput.storage_class == "recovery",
                    Job.execution_state.not_in((
                        ExecutionState.SUCCEEDED.value,
                        ExecutionState.FAILED.value,
                        ExecutionState.CANCELLED.value,
                    )),
                )
            ))
            return paths

    def active_fit_job_ids(self) -> set[str]:
        with self.database.session() as session:
            return set(session.scalars(
                select(Job.job_id).where(
                    Job.operation == "fit",
                    Job.execution_state.not_in((
                        ExecutionState.SUCCEEDED.value,
                        ExecutionState.FAILED.value,
                        ExecutionState.CANCELLED.value,
                    )),
                )
            ))

    def terminal_fit_job_ids(self) -> set[str]:
        with self.database.session() as session:
            return set(session.scalars(
                select(Job.job_id).where(
                    Job.operation == "fit",
                    Job.execution_state.in_((
                        ExecutionState.SUCCEEDED.value,
                        ExecutionState.FAILED.value,
                        ExecutionState.CANCELLED.value,
                    )),
                )
            ))

    def prune_checkpoints(
        self,
        job_id: str,
        *,
        keep: int = 2,
    ) -> list[str]:
        positive(keep, "keep")
        with self.database.transaction() as session:
            rows = session.scalars(
                select(TrainingRecoveryCheckpoint)
                .where(
                    TrainingRecoveryCheckpoint.job_id == job_id
                )
                .order_by(
                    TrainingRecoveryCheckpoint.generation.desc()
                )
                .offset(keep)
                .with_for_update()
            ).all()
            paths = [row.relative_path for row in rows]
            for row in rows:
                session.delete(row)
            return paths

    def schedule_retry(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        error_code: str | ErrorCode,
        error_message: str,
        exit_code: int | None = None,
        now: float | None = None,
    ) -> RowMapping:
        positive(attempt, "attempt")
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        if not error_message:
            raise ValueError("retry error_message must not be empty")
        code = (
            error_code.value
            if isinstance(error_code, ErrorCode)
            else error_code
        )
        if code not in (
            ErrorCode.DEVICE_LOST.value,
            ErrorCode.EXECUTION_INTERRUPTED.value,
            ErrorCode.SUBPROCESS_FAILED.value,
            ErrorCode.SUBPROCESS_HUNG.value,
        ):
            raise ValueError("error_code is not retryable")
        retried_at = timestamp_now(now)
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update()
            )
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job.attempt != attempt:
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            record = session.get(
                JobAttempt,
                (job_id, attempt),
                with_for_update=True,
            )
            if record is None or record.attempt_id != attempt_id:
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            if job.execution_state == ExecutionState.RETRYING.value:
                if record.status != ExecutionState.FAILED.value:
                    raise failed_precondition(
                        "job attempt retry state is inconsistent"
                    )
                return decode(job)
            if (
                job.execution_state != ExecutionState.RUNNING.value
                or (
                    code
                    in (
                        ErrorCode.EXECUTION_INTERRUPTED.value,
                        ErrorCode.SUBPROCESS_FAILED.value,
                        ErrorCode.SUBPROCESS_HUNG.value,
                    )
                    and job.operation != "fit"
                )
            ):
                raise failed_precondition(
                    "job is not retryable for this failure"
                )
            if job.operation == "fit":
                runtime_input = session.scalar(
                    select(JobInput.job_id)
                    .where(
                        JobInput.job_id == job_id,
                        JobInput.storage_class == "runtime",
                    )
                    .limit(1)
                )
                if runtime_input is not None:
                    raise failed_precondition(
                        "fit inputs are not stored for recovery"
                    )
            if (
                record.status != ExecutionState.RUNNING.value
            ):
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            if (
                code == ErrorCode.DEVICE_LOST.value
                and record.selected_device != "cuda"
            ):
                raise failed_precondition(
                    "device loss requires an active CUDA attempt"
                )
            record.status = ExecutionState.FAILED.value
            record.finished_at = retried_at
            record.exit_code = exit_code
            record.error_code = code
            record.error_message = error_message
            job.execution_state = ExecutionState.RETRYING.value
            job.revision += 1
            queue_sequence = session.scalar(
                select(QUEUE_SEQUENCE.next_value())
            )
            if queue_sequence is None:
                raise RuntimeError(
                    "queue sequence did not return a value"
                )
            job.queue_sequence = queue_sequence
            job.queued_at = retried_at
            job.updated_at = retried_at
            job.finished_at = None
            job.error_code = None
            job.error_message = None
            job.waiting_for_input = False
            job.waiting_input_ordinal = None
            job.input_waiting_since = None
            job.acquire_grace_until = None
            session.flush()
            return decode(job)


def _record(
    value: TrainingRecoveryCheckpoint,
) -> TrainingRecoveryCheckpointRecord:
    return TrainingRecoveryCheckpointRecord(
        job_id=value.job_id,
        generation=value.generation,
        attempt=value.attempt,
        format=value.format,
        relative_path=value.relative_path,
        byte_count=value.bytes,
        sha256=value.sha256,
        completed_epochs=value.completed_epochs,
        global_step=value.global_step,
        training_complete=value.training_complete,
    )


def _same_checkpoint(
    value: TrainingRecoveryCheckpoint,
    *,
    attempt: int,
    format: str,
    relative_path: str,
    byte_count: int,
    sha256: str,
    completed_epochs: int,
    global_step: int,
    training_complete: bool,
) -> bool:
    return (
        value.attempt == attempt
        and value.format == format
        and value.relative_path == relative_path
        and value.bytes == byte_count
        and value.sha256 == sha256
        and value.completed_epochs == completed_epochs
        and value.global_step == global_step
        and value.training_complete is training_complete
    )


__all__ = ["RecoveryLedgerSlice"]
