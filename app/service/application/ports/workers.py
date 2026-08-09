from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.service.domain.records import (
    ExecutionJobRecord,
    StagedPredictionOutput,
)


@dataclass(frozen=True)
class ExecutionInput:
    ordinal: int
    schema_id: str
    rows: int
    byte_count: int
    sha256: str
    absolute_path: str
    storage_class: str


@dataclass(frozen=True)
class ExecutionPlan:
    inputs: tuple[ExecutionInput, ...]
    argv: tuple[str, ...]
    uses_spooled_fit: bool
    uses_training_recovery: bool = False
    resume_training_complete: bool = False
    protocol_version: int = 0
    manifest_path: str | None = None
    workspace: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    stderr_tail: bytes
    outputs: tuple[StagedPredictionOutput, ...] = ()
    result_manifest: dict | None = None


class ExecutionPlanBuilder(Protocol):
    """Build one immutable, server-controlled worker command."""

    def build(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        *,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
    ) -> ExecutionPlan: ...


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

    def interrupt_for_shutdown(self) -> None: ...


__all__ = [
    "AttemptProcess",
    "ExecutionInput",
    "ExecutionPlan",
    "ExecutionPlanBuilder",
    "ExecutionResult",
    "WorkerExecutor",
]
