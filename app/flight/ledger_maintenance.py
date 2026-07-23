from __future__ import annotations

from sqlalchemy import delete, select

from app.database.models import (
    IdempotencyRecord,
    InputUpload,
    Job,
    JobAttempt,
    JobInput,
    JobOutput,
    OutputTicket,
    PublishedModel,
    RuntimeState,
)
from app.flight.constants import JobState
from app.flight.ledger_support import (
    at,
    canonical_uuid,
    now as timestamp_now,
)
from app.flight.state import decide_interrupted_attempt

_TERMINAL_STATES = (
    JobState.SUCCEEDED.value,
    JobState.FAILED.value,
    JobState.CANCELLED.value,
)


class MaintenanceLedgerSlice:
    """Restart reconciliation, retention, and runtime epoch operations."""

    def __init__(self, sessions):
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
            return result.rowcount

    def reconcile_interrupted_jobs(
        self,
        *,
        now: float | None = None,
    ) -> dict:
        reconciled_at = timestamp_now(now)
        with self.database.transaction() as session:
            uploads = list(
                session.scalars(select(InputUpload.temporary_path))
            )
            interrupted = list(session.scalars(
                select(Job.job_id)
                .where(Job.state == JobState.RUNNING.value)
                .order_by(Job.job_id)
            ))
            cancelling = list(session.scalars(
                select(Job.job_id)
                .where(Job.state == JobState.CANCELLING.value)
                .order_by(Job.job_id)
            ))
            jobs = session.scalars(
                select(Job)
                .where(Job.state.in_((
                    JobState.RUNNING.value,
                    JobState.CANCELLING.value,
                )))
                .with_for_update()
            )
            for job in jobs:
                attempt = session.get(
                    JobAttempt,
                    (job.job_id, job.attempt),
                    with_for_update=True,
                )
                decision = decide_interrupted_attempt(job.state)
                job.state = decision.target.value
                if decision.error_code is not None:
                    job.error_code = decision.error_code.value
                    job.error_message = decision.error_message
                if decision.target == JobState.FAILED:
                    if (
                        attempt is not None
                        and attempt.status == JobState.RUNNING.value
                    ):
                        attempt.status = decision.target.value
                        attempt.error_code = job.error_code
                        attempt.error_message = job.error_message
                        attempt.finished_at = reconciled_at
                elif (
                    attempt is not None
                    and attempt.status == JobState.RUNNING.value
                ):
                    attempt.status = decision.target.value
                    attempt.finished_at = reconciled_at
                job.revision += 1
                job.finished_at = reconciled_at
                job.updated_at = reconciled_at
            session.execute(delete(InputUpload))
        return {
            "interrupted_jobs": interrupted,
            "cancelled_jobs": cancelling,
            "temporary_paths": uploads,
        }

    def referenced_paths(self) -> set[str]:
        with self.database.session() as session:
            paths = set(
                session.scalars(select(JobInput.relative_path))
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
                    Job.state.in_(_TERMINAL_STATES),
                    Job.finished_at.is_not(None),
                    Job.finished_at < cutoff_at,
                    ~select(PublishedModel.model_ref)
                    .where(
                        PublishedModel.producing_job_id == Job.job_id
                    )
                    .exists(),
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
    ) -> dict:
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
            discarded_jobs = list(
                session.scalars(
                    select(Job.job_id).order_by(Job.job_id)
                )
            )
            reset = state is not None or bool(discarded_jobs)
            session.execute(delete(OutputTicket))
            session.execute(delete(IdempotencyRecord))
            session.execute(delete(InputUpload))
            session.execute(delete(Job))
            if state is None:
                session.add(RuntimeState(
                    key="storage_epoch",
                    value=epoch,
                    updated_at=synchronized_at,
                ))
            else:
                state.value = epoch
                state.updated_at = synchronized_at
            return {
                "reset": reset,
                "discarded_jobs": discarded_jobs,
            }


__all__ = ["MaintenanceLedgerSlice"]
