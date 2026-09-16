from __future__ import annotations

from datetime import datetime
from typing import Protocol, cast

from sqlalchemy import delete, or_, select, update

from app.contracts.json_types import JsonObject
from app.service.adapters.outbound.postgres.ledger.support import (
    LedgerSessions,
    at,
    canonical_uuid,
    json_value,
    now as timestamp_now,
)
from app.service.adapters.outbound.postgres.models import (
    IdempotencyRecord,
    InputUpload,
    Job,
    JobAttempt,
    JobIdentity,
    JobInput,
    JobOutput,
    OutputTicket,
    RuntimeState,
)
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.policies import decide_startup_interruption

_TERMINAL_STATES = (
    ExecutionState.SUCCEEDED.value,
    ExecutionState.FAILED.value,
    ExecutionState.CANCELLED.value,
)


class _RowCountResult(Protocol):
    rowcount: int


class MaintenanceLedgerSlice:
    """Restart reconciliation, retention, and runtime epoch operations."""

    def __init__(self, sessions: LedgerSessions) -> None:
        self.sessions = sessions
        self.database = sessions.database

    def delete_expired_tickets(
        self,
        *,
        now: float | None = None,
    ) -> int:
        with self.database.transaction() as session:
            result = session.execute(
                delete(OutputTicket).where(
                    OutputTicket.expires_at <= timestamp_now(now)
                )
            )
            return cast(_RowCountResult, result).rowcount

    def expire_input_waits(
        self,
        *,
        timeout_seconds: float,
        now: float | None = None,
    ) -> list[str]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        expired_at = timestamp_now(now)
        cutoff = datetime.fromtimestamp(
            expired_at.timestamp() - timeout_seconds,
            tz=expired_at.tzinfo,
        )
        expired: list[str] = []
        with self.database.transaction() as session:
            jobs = session.scalars(
                select(Job)
                .where(
                    Job.waiting_for_input.is_(True),
                    Job.input_state == InputState.OPEN.value,
                    Job.execution_state == ExecutionState.RUNNING.value,
                    Job.input_waiting_since <= cutoff,
                    or_(
                        Job.acquire_grace_until.is_(None),
                        Job.acquire_grace_until <= expired_at,
                    ),
                )
                .order_by(Job.job_id)
                .with_for_update(skip_locked=True)
            )
            for job in jobs:
                attempt = session.get(
                    JobAttempt,
                    (job.job_id, job.attempt),
                    with_for_update=True,
                )
                if (
                    attempt is not None
                    and attempt.status == ExecutionState.RUNNING.value
                ):
                    attempt.status = ExecutionState.FAILED.value
                    attempt.error_code = ErrorCode.INPUT_TIMEOUT.value
                    attempt.error_message = (
                        "input stream timed out while waiting for the next ordinal"
                    )
                    attempt.finished_at = expired_at
                job.input_state = InputState.ABORTED.value
                job.execution_state = ExecutionState.FAILED.value
                job.error_code = ErrorCode.INPUT_TIMEOUT.value
                job.error_message = (
                    "input stream timed out while waiting for the next ordinal"
                )
                job.waiting_for_input = False
                job.waiting_input_ordinal = None
                job.input_waiting_since = None
                job.acquire_grace_until = None
                job.finished_at = expired_at
                job.updated_at = expired_at
                job.revision += 1
                expired.append(job.job_id)
        return expired

    def reconcile_startup_jobs(
        self,
        *,
        now: float | None = None,
    ) -> JsonObject:
        reconciled_at = timestamp_now(now)
        with self.database.transaction() as session:
            uploads = session.execute(
                select(
                    InputUpload.candidate_path,
                    InputUpload.storage_class,
                )
            ).all()
            failed_waiting_input: list[str] = []
            failed_queued: list[str] = []
            failed_running: list[str] = []
            failed_retrying: list[str] = []
            cancelled: list[str] = []
            jobs = session.scalars(
                select(Job)
                .where(Job.execution_state.in_((
                    ExecutionState.WAITING_INPUT.value,
                    ExecutionState.QUEUED.value,
                    ExecutionState.RUNNING.value,
                    ExecutionState.RETRYING.value,
                    ExecutionState.CANCELLING.value,
                )))
                .order_by(Job.job_id)
                .with_for_update()
            )
            for job in jobs:
                current_state = ExecutionState(job.execution_state)
                attempt = session.get(
                    JobAttempt,
                    (job.job_id, job.attempt),
                    with_for_update=True,
                )
                decision = decide_startup_interruption(current_state)
                job.execution_state = decision.target.value
                if (
                    decision.target
                    in (ExecutionState.FAILED, ExecutionState.CANCELLED)
                    and job.input_state == InputState.OPEN.value
                ):
                    job.input_state = InputState.ABORTED.value
                if decision.error_code is not None:
                    job.error_code = decision.error_code.value
                    job.error_message = decision.error_message
                if decision.target == ExecutionState.FAILED:
                    if (
                        attempt is not None
                        and attempt.status == ExecutionState.RUNNING.value
                    ):
                        attempt.status = decision.target.value
                        attempt.error_code = job.error_code
                        attempt.error_message = job.error_message
                        attempt.finished_at = reconciled_at
                elif (
                    attempt is not None
                    and attempt.status == ExecutionState.RUNNING.value
                ):
                    attempt.status = decision.target.value
                    attempt.finished_at = reconciled_at
                job.revision += 1
                job.finished_at = reconciled_at
                job.updated_at = reconciled_at
                if decision.target == ExecutionState.CANCELLED:
                    cancelled.append(job.job_id)
                elif current_state == ExecutionState.WAITING_INPUT:
                    failed_waiting_input.append(job.job_id)
                elif current_state == ExecutionState.QUEUED:
                    failed_queued.append(job.job_id)
                elif current_state == ExecutionState.RUNNING:
                    failed_running.append(job.job_id)
                elif current_state == ExecutionState.RETRYING:
                    failed_retrying.append(job.job_id)
                else:
                    raise AssertionError(
                        "unexpected startup reconciliation state"
                    )
            session.execute(delete(InputUpload))
        return json_value({
            "failed_waiting_input_jobs": failed_waiting_input,
            "failed_queued_jobs": failed_queued,
            "failed_running_jobs": failed_running,
            "failed_retrying_jobs": failed_retrying,
            "cancelled_jobs": cancelled,
            "temporary_paths": [
                path
                for path, storage_class in uploads
                if storage_class == "runtime"
            ],
            "recovery_temporary_paths": [
                path
                for path, storage_class in uploads
                if storage_class == "recovery"
            ],
        })

    def referenced_paths(self) -> set[str]:
        with self.database.session() as session:
            paths = set(
                session.scalars(
                    select(JobInput.relative_path).where(
                        JobInput.storage_class == "runtime"
                    )
                )
            )
            paths.update(
                session.scalars(select(JobOutput.relative_path))
            )
            return paths

    def delete_terminal_jobs_before(self, cutoff: float) -> list[str]:
        cutoff_at = at(cutoff)
        with self.database.transaction() as session:
            candidates = session.scalars(
                select(Job)
                .where(
                    Job.execution_state.in_(_TERMINAL_STATES),
                    Job.finished_at.is_not(None),
                    Job.finished_at < cutoff_at,
                    ~select(OutputTicket.ticket_hash)
                    .where(OutputTicket.job_id == Job.job_id)
                    .exists(),
                    ~select(IdempotencyRecord.idempotency_key)
                    .where(
                        IdempotencyRecord.job_id == Job.job_id,
                        IdempotencyRecord.created_at >= cutoff_at,
                    )
                    .exists(),
                )
                .order_by(Job.job_id)
                .with_for_update()
            ).all()
            job_ids = [job.job_id for job in candidates]
            if job_ids:
                session.execute(
                    delete(IdempotencyRecord).where(
                        IdempotencyRecord.job_id.in_(job_ids)
                    )
                )
                session.execute(
                    delete(Job).where(Job.job_id.in_(job_ids))
                )
                session.execute(
                    update(JobIdentity)
                    .where(JobIdentity.job_id.in_(job_ids))
                    .values(retired_at=cutoff_at)
                )
            session.execute(
                delete(IdempotencyRecord).where(
                    IdempotencyRecord.job_id.is_(None),
                    IdempotencyRecord.created_at < cutoff_at,
                )
            )
            return job_ids

    def synchronize_runtime_epoch(
        self,
        epoch: str,
        *,
        now: float | None = None,
    ) -> JsonObject:
        epoch = canonical_uuid(epoch, "runtime storage epoch")
        synchronized_at = timestamp_now(now)
        with self.database.transaction() as session:
            state = session.get(
                RuntimeState,
                "storage_epoch",
                with_for_update=True,
            )
            if state is not None and state.value == epoch:
                return {"reset": False, "discarded_jobs": []}
            discarded_jobs: list[str] = []
            jobs = session.scalars(
                select(Job).order_by(Job.job_id)
            ).all()
            for job in jobs:
                if job.operation != "fit":
                    discarded_jobs.append(job.job_id)
                    continue
                runtime_input = session.scalar(
                    select(JobInput.job_id)
                    .where(
                        JobInput.job_id == job.job_id,
                        JobInput.storage_class == "runtime",
                    )
                    .limit(1)
                )
                if runtime_input is not None:
                    discarded_jobs.append(job.job_id)
            reset = state is not None or bool(discarded_jobs)
            if discarded_jobs:
                session.execute(
                    delete(IdempotencyRecord).where(
                        IdempotencyRecord.job_id.in_(
                            discarded_jobs
                        )
                    )
                )
                session.execute(
                    delete(Job).where(
                        Job.job_id.in_(discarded_jobs)
                    )
                )
                session.execute(
                    update(JobIdentity)
                    .where(JobIdentity.job_id.in_(discarded_jobs))
                    .values(retired_at=synchronized_at)
                )
            if state is None:
                session.add(RuntimeState(
                    key="storage_epoch",
                    value=epoch,
                    updated_at=synchronized_at,
                ))
            else:
                state.value = epoch
                state.updated_at = synchronized_at
            return json_value({
                "reset": reset,
                "discarded_jobs": discarded_jobs,
            })


__all__ = ["MaintenanceLedgerSlice"]
