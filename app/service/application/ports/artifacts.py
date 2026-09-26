from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.service.application.ports.workers import ExecutionInput
from app.service.domain.json_types import JsonObject
from app.service.domain.records import ExecutionJobRecord


@dataclass(frozen=True, slots=True)
class PublishedModelArtifacts:
    """Результат публикации ядра модели, доступный необязательному наблюдателю."""

    model_ref: str
    model_contract: JsonObject
    checkpoint_publication_ms: float


class ArtifactPublisher(Protocol):
    """Проверить и опубликовать артефакты, созданные одной активной попыткой."""

    def publish_outputs_from_manifest(
        self,
        job: ExecutionJobRecord,
        inputs: tuple[ExecutionInput, ...],
        result: JsonObject,
    ) -> None: ...

    def publish_model_from_manifest(
        self,
        job: ExecutionJobRecord,
        result: JsonObject,
    ) -> PublishedModelArtifacts: ...

    def cleanup_unpublished(self, job: ExecutionJobRecord) -> None: ...


__all__ = ["ArtifactPublisher", "PublishedModelArtifacts"]
