from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.service.domain.json_types import JsonObject
from app.service.domain.records import MetricsOutboxRecord


class RetryableMetricsDeliveryError(RuntimeError):
    pass


class BlockedMetricsDeliveryError(RuntimeError):
    pass


class MetricsOutboxRepository(Protocol):
    def next_pending(self) -> MetricsOutboxRecord | None: ...

    def advance(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
        cursor: int,
    ) -> bool: ...

    def retry(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
        delay_seconds: float,
        error_code: str,
        error_message: str,
    ) -> bool: ...

    def block(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
        error_code: str,
        error_message: str,
    ) -> bool: ...

    def complete(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
    ) -> bool: ...

    def purge_delivered(self, *, older_than_seconds: float) -> int: ...

    def backlog(self) -> tuple[int, int, float | None]: ...


class MetricsArtifactProjection(Protocol):
    def points(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> tuple[JsonObject, ...]: ...

    def artifact_document(
        self,
        entry: MetricsOutboxRecord,
        *,
        deployment_id: str,
    ) -> JsonObject: ...


class MetricsDocumentSink(Protocol):
    def create_documents(
        self,
        stream: str,
        documents: Sequence[JsonObject],
        *,
        id_field: str,
    ) -> None: ...


__all__ = [
    "BlockedMetricsDeliveryError",
    "MetricsArtifactProjection",
    "MetricsDocumentSink",
    "MetricsOutboxRepository",
    "RetryableMetricsDeliveryError",
]
