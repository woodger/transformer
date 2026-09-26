from __future__ import annotations

from app.service.application.ports.models import ModelAdministrationStore
from app.service.domain.records import ModelLifecycleRecord


class ModelAdministration:
    """Административные сценарии для опубликованных поколений моделей."""

    def __init__(self, store: ModelAdministrationStore) -> None:
        self.store = store

    def list(self, *, deleted: bool = False) -> list[ModelLifecycleRecord]:
        return self.store.list_models(deleted=deleted)

    def delete(self, model_ref: str) -> ModelLifecycleRecord:
        return self.store.request_deletion(model_ref)


__all__ = ["ModelAdministration"]
