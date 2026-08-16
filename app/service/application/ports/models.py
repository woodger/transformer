from __future__ import annotations

from typing import Protocol

from app.service.domain.records import ModelLifecycleRecord


class ModelAdministrationStore(Protocol):
    def list_models(self) -> list[ModelLifecycleRecord]: ...

    def request_deletion(
        self,
        model_ref: str,
        *,
        discard_undelivered_metrics: bool,
    ) -> ModelLifecycleRecord: ...


class ModelDeletionRepository(Protocol):
    def pending_deletions(self, *, limit: int = 100) -> tuple[str, ...]: ...

    def complete_deletion(self, model_ref: str) -> bool: ...


__all__ = ["ModelAdministrationStore", "ModelDeletionRepository"]
