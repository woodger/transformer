from __future__ import annotations

from typing import Protocol

from app.service.application.ports.workers import ExecutionInput
from app.service.domain.records import ExecutionJobRecord


class ArtifactPublisher(Protocol):
    """Validate and publish artifacts produced by one active attempt."""

    def publish_outputs_from_manifest(
        self,
        job: ExecutionJobRecord,
        inputs: tuple[ExecutionInput, ...],
        result: dict,
    ) -> None: ...

    def publish_model_from_manifest(
        self,
        job: ExecutionJobRecord,
        result: dict,
    ) -> None: ...

    def cleanup_unpublished(self, job: ExecutionJobRecord) -> None: ...


__all__ = ["ArtifactPublisher"]
