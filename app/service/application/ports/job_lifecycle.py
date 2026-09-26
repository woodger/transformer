from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from app.service.application.messages.jobs import (
    AcquireJobCommand,
    CancelJobCommand,
    CloseInputCommand,
    CreateJobCommand,
    InputClosed,
    JobAcquired,
    JobCancelled,
    JobCreated,
    JobCreationPreparation,
)
from app.service.domain.records import PublishedModelRecord

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ArtifactLocation:
    storage_class: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class LifecycleMutation(Generic[ResultT]):
    result: ResultT
    replayed: bool
    cleanup: tuple[ArtifactLocation, ...] = ()
    queued: bool = False


class JobLifecycleStore(Protocol):
    """Операции атомарного хранения, нужные публичным изменениям job."""

    def create(
        self,
        command: CreateJobCommand,
        *,
        max_active_jobs: int,
        preflight: Callable[[], None],
        prepare: Callable[
            [PublishedModelRecord | None],
            JobCreationPreparation,
        ],
    ) -> LifecycleMutation[JobCreated]: ...

    def acquire(
        self,
        command: AcquireJobCommand,
        *,
        acquire_grace_seconds: float,
    ) -> LifecycleMutation[JobAcquired]: ...

    def close_input(
        self,
        command: CloseInputCommand,
        *,
        select_device: Callable[[str, str | None, str, int], str],
    ) -> LifecycleMutation[InputClosed]: ...

    def cancel(
        self,
        command: CancelJobCommand,
        *,
        stop_active_worker: Callable[[str], None],
    ) -> LifecycleMutation[JobCancelled]: ...


class CandidateArtifactCleaner(Protocol):
    def cleanup(self, location: ArtifactLocation) -> None: ...


class ModelArtifactVerifier(Protocol):
    def verify(self, model: PublishedModelRecord) -> None: ...


__all__ = [
    "ArtifactLocation",
    "CandidateArtifactCleaner",
    "JobLifecycleStore",
    "LifecycleMutation",
    "ModelArtifactVerifier",
]
