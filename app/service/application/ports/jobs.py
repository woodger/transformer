from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.service.domain.job import ErrorCode, JobState
from app.service.domain.records import (
    CommittedInputRecord,
    ExecutionJobRecord,
    TrainingRecoveryCheckpointRecord,
)


class JobRepository(Protocol):
    """Durable job capabilities required by execution orchestration."""

    def get_execution_job(self, job_id: str) -> ExecutionJobRecord | None: ...

    def queued_execution_jobs(self) -> list[ExecutionJobRecord]: ...

    def claim_execution_job(
        self,
        job_id: str,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
    ) -> ExecutionJobRecord | None: ...

    def list_committed_inputs(
        self,
        job_id: str,
    ) -> Sequence[CommittedInputRecord]: ...

    def update_progress(
        self,
        job_id: str,
        progress: dict,
        *,
        attempt_id: str,
    ) -> dict: ...

    def finish_attempt(
        self,
        job_id: str,
        attempt: int,
        target_state: JobState,
        *,
        attempt_id: str,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
        exit_code: int | None = None,
    ) -> dict: ...

    def latest_recovery_checkpoint(
        self,
        job_id: str,
    ) -> TrainingRecoveryCheckpointRecord | None: ...

    def request_attempt_cancel(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
    ) -> bool: ...

    def schedule_retry(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        error_code: ErrorCode,
        error_message: str,
        exit_code: int | None = None,
    ) -> dict: ...


__all__ = ["JobRepository"]
