from __future__ import annotations

import uuid
from datetime import datetime
from typing import cast, overload

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.contracts.json_types import JsonObject
from app.contracts.worker.v6.config import ModelConfig, TrainConfig
from app.service.adapters.outbound.postgres.ledger.support import (
    LedgerSessions,
    RowMapping,
    canonical_uuid,
    decode,
    json_value,
    now as timestamp_now,
    positive,
    timestamp,
)
from app.service.adapters.outbound.postgres.models import (
    Job,
    JobAttempt,
    TrainingRecoveryCheckpoint,
)
from app.service.domain.errors import failed_precondition, not_found
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.policies import validate_execution_transition
from app.service.domain.records import (
    ExecutionJobRecord,
    RecoverableAttemptRecord,
)


class ExecutionLedgerSlice:
    """PostgreSQL queue, attempt, and worker-execution operations."""

    def __init__(self, sessions: LedgerSessions) -> None:
        self.sessions = sessions
        self.database = sessions.database

    def get_execution_job(
        self,
        job_id: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
        for_update: bool = False,
    ) -> ExecutionJobRecord | None:
        statement = select(Job).where(Job.job_id == job_id)
        if owner_subject is not None:
            statement = statement.where(
                Job.owner_subject == owner_subject
            )
        if for_update:
            statement = statement.with_for_update()
        with self.sessions.read(connection) as session:
            job = session.scalar(statement)
            attempt = (
                None
                if job is None or job.attempt <= 0
                else session.get(
                    JobAttempt,
                    (job.job_id, job.attempt),
                )
            )
            return _execution_job_record(job, attempt)

    def update_progress(
        self,
        job_id: str,
        progress: JsonObject,
        *,
        attempt_id: str,
        now: float | None = None,
    ) -> RowMapping:
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update()
            )
            if (
                job is None
                or job.execution_state != ExecutionState.RUNNING.value
            ):
                raise failed_precondition("job is not running")
            record = session.get(
                JobAttempt,
                (job_id, job.attempt),
                with_for_update=True,
            )
            if (
                record is None
                or record.status != ExecutionState.RUNNING.value
                or record.attempt_id != attempt_id
            ):
                raise failed_precondition("job attempt is no longer active")
            job.progress = json_value(progress)
            job.revision += 1
            job.updated_at = timestamp_now(now)
            session.flush()
            return decode(job)

    def claim_job(
        self,
        job_id: str,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        now: float | None = None,
    ) -> RowMapping | None:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update(skip_locked=True)
            )
            if (
                job is None
                or job.execution_state not in (
                    ExecutionState.QUEUED.value,
                    ExecutionState.RETRYING.value,
                )
                or job.selected_device != selected_device
            ):
                return None
            job, attempt = self._claim(
                session,
                job,
                worker_id,
                timestamp_now(now),
                device_id=device_id,
            )
            return _claimed_job_mapping(job, attempt)

    def claim_execution_job(
        self,
        job_id: str,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        now: float | None = None,
    ) -> ExecutionJobRecord | None:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update(skip_locked=True)
            )
            if (
                job is None
                or job.execution_state not in (
                    ExecutionState.QUEUED.value,
                    ExecutionState.RETRYING.value,
                )
                or job.selected_device != selected_device
            ):
                return None
            job, attempt = self._claim(
                session,
                job,
                worker_id,
                timestamp_now(now),
                device_id=device_id,
            )
            return _execution_job_record(job, attempt)

    def claim_next_job(
        self,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        now: float | None = None,
    ) -> RowMapping | None:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(
                    Job.execution_state.in_((
                        ExecutionState.QUEUED.value,
                        ExecutionState.RETRYING.value,
                    )),
                    Job.selected_device == selected_device,
                )
                .order_by(Job.queue_sequence)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            job, attempt = self._claim(
                session,
                job,
                worker_id,
                timestamp_now(now),
                device_id=device_id,
            )
            return _claimed_job_mapping(job, attempt)

    def _claim(
        self,
        session: Session,
        job: Job,
        worker_id: str | None,
        claimed_at: datetime,
        *,
        device_id: str | None = None,
    ) -> tuple[Job, JobAttempt]:
        if job.selected_device == "cuda" and not device_id:
            raise ValueError(
                "CUDA attempt requires a physical device assignment"
            )
        if job.selected_device == "cpu" and device_id is not None:
            raise ValueError(
                "CPU attempt must not carry a CUDA device assignment"
            )
        resume_generation = None
        if job.operation == "fit":
            resume_generation = session.scalar(
                select(TrainingRecoveryCheckpoint.generation)
                .where(
                    TrainingRecoveryCheckpoint.job_id == job.job_id
                )
                .order_by(
                    TrainingRecoveryCheckpoint.generation.desc()
                )
                .limit(1)
            )
        job.execution_state = ExecutionState.RUNNING.value
        job.revision += 1
        job.attempt += 1
        job.started_at = claimed_at
        job.updated_at = claimed_at
        record = JobAttempt(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=str(uuid.uuid4()),
            selected_device=job.selected_device,
            device_id=device_id,
            resume_generation=resume_generation,
            status=ExecutionState.RUNNING.value,
            worker_id=worker_id,
            claimed_at=claimed_at,
            queue_entered_at=job.queued_at or claimed_at,
            started_at=claimed_at,
        )
        session.add(record)
        session.flush()
        return job, record

    def set_attempt_process(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        pid: int,
        pgid: int,
        boot_id: str,
        process_start_ticks: int,
    ) -> None:
        positive(pid, "pid")
        positive(pgid, "pgid")
        positive(process_start_ticks, "process_start_ticks")
        boot_id = canonical_uuid(boot_id, "boot_id")
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        with self.database.transaction() as session:
            record = session.get(
                JobAttempt,
                (job_id, attempt),
                with_for_update=True,
            )
            if (
                record is None
                or record.status != ExecutionState.RUNNING.value
                or record.attempt_id != attempt_id
            ):
                raise failed_precondition("job attempt is not running")
            record.pid = pid
            record.pgid = pgid
            record.boot_id = boot_id
            record.process_start_ticks = process_start_ticks

    def mark_attempt_worker_ready(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        now: float | None = None,
    ) -> None:
        self._mark_attempt_boundary(
            job_id,
            attempt,
            attempt_id=attempt_id,
            field="worker_ready_at",
            now=now,
        )

    def mark_attempt_worker_completed(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        now: float | None = None,
    ) -> None:
        self._mark_attempt_boundary(
            job_id,
            attempt,
            attempt_id=attempt_id,
            field="worker_completed_at",
            now=now,
        )

    def _mark_attempt_boundary(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        field: str,
        now: float | None,
    ) -> None:
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        measured_at = timestamp_now(now)
        with self.database.transaction() as session:
            record = session.get(
                JobAttempt,
                (job_id, attempt),
                with_for_update=True,
            )
            if (
                record is None
                or record.status != ExecutionState.RUNNING.value
                or record.attempt_id != attempt_id
            ):
                raise failed_precondition("job attempt is not running")
            if getattr(record, field) is None:
                setattr(record, field, measured_at)

    def list_active_attempts(self) -> list[RowMapping]:
        with self.database.session() as session:
            rows = session.scalars(
                select(JobAttempt)
                .join(Job, Job.job_id == JobAttempt.job_id)
                .where(
                    Job.execution_state.in_((
                        ExecutionState.RUNNING.value,
                        ExecutionState.CANCELLING.value,
                    )),
                    JobAttempt.status == ExecutionState.RUNNING.value,
                    JobAttempt.attempt == Job.attempt,
                )
                .order_by(JobAttempt.job_id, JobAttempt.attempt)
            )
            return [decode(row) for row in rows]

    def list_recoverable_attempts(
        self,
    ) -> list[RecoverableAttemptRecord]:
        with self.database.session() as session:
            rows = session.scalars(
                select(JobAttempt)
                .join(Job, Job.job_id == JobAttempt.job_id)
                .where(
                    Job.execution_state.in_((
                        ExecutionState.RUNNING.value,
                        ExecutionState.CANCELLING.value,
                    )),
                    JobAttempt.status == ExecutionState.RUNNING.value,
                    JobAttempt.attempt == Job.attempt,
                )
                .order_by(JobAttempt.job_id, JobAttempt.attempt)
            )
            return [_recoverable_attempt_record(row) for row in rows]

    def finish_attempt(
        self,
        job_id: str,
        attempt: int,
        target_state: str | ExecutionState,
        *,
        attempt_id: str,
        error_code: str | ErrorCode | None = None,
        error_message: str | None = None,
        exit_code: int | None = None,
        now: float | None = None,
    ) -> RowMapping:
        target_state = ExecutionState(target_state)
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        if target_state not in (
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        ):
            raise ValueError(
                "worker attempt can finish only as FAILED or CANCELLED"
            )
        if target_state == ExecutionState.CANCELLED and (
            error_code is not None or error_message is not None
        ):
            raise ValueError("cancelled attempt must not carry an error")
        finished_at = timestamp_now(now)
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
            validate_execution_transition(job.execution_state, target_state)
            record = session.get(
                JobAttempt,
                (job_id, attempt),
                with_for_update=True,
            )
            if (
                record is None
                or record.status != ExecutionState.RUNNING.value
                or record.attempt_id != attempt_id
            ):
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            code = (
                error_code.value
                if isinstance(error_code, ErrorCode)
                else error_code
            )
            record.status = target_state.value
            record.finished_at = finished_at
            record.exit_code = exit_code
            record.error_code = code
            record.error_message = error_message
            job.execution_state = target_state.value
            if job.input_state == InputState.OPEN.value:
                job.input_state = InputState.ABORTED.value
            job.revision += 1
            job.error_code = code
            job.error_message = error_message
            job.finished_at = finished_at
            job.updated_at = finished_at
            session.flush()
            return decode(job)

    def request_attempt_cancel(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        now: float | None = None,
    ) -> bool:
        """Move the active attempt to CANCELLING under its equality fence."""
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        requested_at = timestamp_now(now)
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
            if (
                record is None
                or record.status != ExecutionState.RUNNING.value
                or record.attempt_id != attempt_id
            ):
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            if job.execution_state == ExecutionState.CANCELLING.value:
                return False
            if job.execution_state != ExecutionState.RUNNING.value:
                raise failed_precondition("job attempt is not running")
            validate_execution_transition(
                job.execution_state,
                ExecutionState.CANCELLING,
            )
            job.execution_state = ExecutionState.CANCELLING.value
            job.revision += 1
            job.cancel_requested_at = requested_at
            job.updated_at = requested_at
            session.flush()
            return True

    def mark_input_waiting(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        next_ordinal: int,
        input_revision: int,
        now: float | None = None,
    ) -> bool:
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        for value in (next_ordinal, input_revision):
            raw_value = cast(object, value)
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise ValueError(
                    "input wait ordinal and revision must be non-negative"
                )
        waiting_at = timestamp_now(now)
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update()
            )
            record = session.get(
                JobAttempt,
                (job_id, attempt),
                with_for_update=True,
            )
            if (
                job is None
                or record is None
                or job.attempt != attempt
                or record.attempt_id != attempt_id
                or record.status != ExecutionState.RUNNING.value
                or job.execution_state != ExecutionState.RUNNING.value
            ):
                raise failed_precondition(
                    "job attempt is no longer active"
                )
            if job.input_state != InputState.OPEN.value:
                return False
            if (
                job.next_input_ordinal != next_ordinal
                or input_revision > job.input_revision
            ):
                return False
            job.waiting_for_input = True
            job.waiting_input_ordinal = next_ordinal
            job.input_waiting_since = waiting_at
            if (
                job.acquire_grace_until is not None
                and job.acquire_grace_until <= waiting_at
            ):
                job.acquire_grace_until = None
            session.flush()
            return True

    def queued_jobs(self) -> list[RowMapping]:
        with self.database.session() as session:
            rows = session.scalars(
                select(Job)
                .where(Job.execution_state.in_((
                    ExecutionState.QUEUED.value,
                    ExecutionState.RETRYING.value,
                )))
                .order_by(Job.queue_sequence)
            )
            return [decode(row) for row in rows]

    def queued_execution_jobs(self) -> list[ExecutionJobRecord]:
        with self.database.session() as session:
            rows = session.scalars(
                select(Job)
                .where(Job.execution_state.in_((
                    ExecutionState.QUEUED.value,
                    ExecutionState.RETRYING.value,
                )))
                .order_by(Job.queue_sequence)
            )
            return [_execution_job_record(row) for row in rows]


@overload
def _execution_job_record(
    record: None,
    attempt: JobAttempt | None = None,
) -> None: ...


@overload
def _execution_job_record(
    record: Job,
    attempt: JobAttempt | None = None,
) -> ExecutionJobRecord: ...


def _execution_job_record(
    record: Job | None,
    attempt: JobAttempt | None = None,
) -> ExecutionJobRecord | None:
    if record is None:
        return None
    return ExecutionJobRecord(
        job_id=record.job_id,
        owner_subject=record.owner_subject,
        operation=record.operation,
        input_state=InputState(record.input_state),
        execution_state=ExecutionState(record.execution_state),
        input_revision=record.input_revision,
        selected_device=record.selected_device,
        model_label=record.model_label,
        input_model_ref=record.resolved_model_ref,
        prediction_column=record.prediction_column,
        model_config=ModelConfig.from_dict(record.model_config),
        training_config=TrainConfig.from_dict(record.training_config),
        data_contract=json_value(record.data_contract),
        ml_contract=json_value(record.ml_contract),
        config_hash=record.config_hash,
        manifest_sha256=record.manifest_sha256,
        feature_dim=record.feature_dim,
        input_frame_count=record.next_input_ordinal,
        attempt=record.attempt,
        assigned_device_id=(
            None if attempt is None else attempt.device_id
        ),
        resume_generation=(
            None if attempt is None else attempt.resume_generation
        ),
        queued_at=timestamp(record.queued_at),
        started_at=timestamp(record.started_at),
        attempt_id=(None if attempt is None else attempt.attempt_id),
    )


def _claimed_job_mapping(job: Job, attempt: JobAttempt) -> RowMapping:
    result = decode(job)
    result["attempt_id"] = attempt.attempt_id
    return result


def _recoverable_attempt_record(
    record: JobAttempt,
) -> RecoverableAttemptRecord:
    return RecoverableAttemptRecord(
        job_id=record.job_id,
        attempt=record.attempt,
        pid=record.pid,
        pgid=record.pgid,
        boot_id=record.boot_id,
        process_start_ticks=record.process_start_ticks,
        attempt_id=record.attempt_id,
    )


__all__ = ["ExecutionLedgerSlice"]
