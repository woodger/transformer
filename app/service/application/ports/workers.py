from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from app.service.domain.records import ExecutionJobRecord


@dataclass(frozen=True)
class ExecutionInput:
    ordinal: int
    commit_revision: int
    schema_id: str
    data_contract_sha256: str
    rows: int
    byte_count: int
    sha256: str
    absolute_path: str
    storage_class: str


@dataclass(frozen=True)
class ExecutionPlan:
    inputs: tuple[ExecutionInput, ...]
    argv: tuple[str, ...]
    protocol_version: int
    manifest_path: str
    workspace: str


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    stderr_tail: bytes
    result_manifest: dict


class ExecutionPlanBuilder(Protocol):
    """Build one immutable, server-controlled worker command."""

    def build(
        self,
        job: ExecutionJobRecord,
        attempt: int,
    ) -> ExecutionPlan: ...

    def streaming_inputs(
        self,
        job: ExecutionJobRecord,
        start_ordinal: int,
    ) -> tuple[ExecutionInput, ...]: ...


class AttemptProcess(Protocol):
    """Run and fully reap one process-contract execution."""

    def run(
        self,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        *,
        cancel: threading.Event,
        force_stop: threading.Event,
    ) -> ExecutionResult: ...


class WorkerExecutor(Protocol):
    """Start and supervise one isolated execution attempt."""

    def execute(self, job: ExecutionJobRecord) -> None: ...

    def notify_cancel(self, job_id: str) -> None: ...

    def notify_input(self, job_id: str) -> None: ...

    def interrupt_for_shutdown(self) -> None: ...


__all__ = [
    "AttemptProcess",
    "ExecutionInput",
    "ExecutionPlan",
    "ExecutionPlanBuilder",
    "ExecutionResult",
    "WorkerExecutor",
]
