from __future__ import annotations

from typing import Protocol

from app.service.application.ports.operations import DiskUsage


class MaintenanceRepository(Protocol):
    def delete_expired_tickets(self, *, now: float) -> int: ...

    def expire_input_waits(
        self,
        *,
        timeout_seconds: float,
        now: float,
    ) -> list[str]: ...

    def delete_terminal_jobs_before(self, cutoff: float) -> list[str]: ...

    def terminal_recovery_job_ids(self) -> set[str]: ...


class RetentionArtifactStore(Protocol):
    def disk_usage(self) -> DiskUsage: ...

    def job_directory(self, job_id: str) -> str: ...

    def remove(self, path: str) -> bool: ...


__all__ = ["MaintenanceRepository", "RetentionArtifactStore"]
