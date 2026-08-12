from __future__ import annotations

from typing import Protocol

from app.service.domain.job import ExecutionState
from app.service.domain.records import OutputRecord


class OutputAccessStore(Protocol):
    def execution_state(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> ExecutionState | None: ...

    def find_output(
        self,
        job_id: str,
        ordinal: int,
    ) -> OutputRecord | None: ...

    def issue_ticket(
        self,
        *,
        job_id: str,
        ordinal: int,
        owner_subject: str,
        ttl_seconds: float,
    ) -> tuple[bytes, float]: ...

    def resolve_ticket(
        self,
        ticket: bytes,
        *,
        owner_subject: str,
    ) -> OutputRecord: ...


class OutputArtifactStore(Protocol):
    def absolute_path(self, relative_path: str) -> str: ...


__all__ = ["OutputAccessStore", "OutputArtifactStore"]
