from __future__ import annotations

from app.service.application.ports.models import ModelAdministrationStore
from app.service.domain.records import ModelLifecycleRecord


class ModelAdministration:
    """Administrative use cases for published model generations."""

    def __init__(self, store: ModelAdministrationStore) -> None:
        self.store = store

    def list(self) -> list[ModelLifecycleRecord]:
        return self.store.list_models()

    def delete(
        self,
        model_ref: str,
        *,
        discard_undelivered_metrics: bool = False,
    ) -> ModelLifecycleRecord:
        return self.store.request_deletion(
            model_ref,
            discard_undelivered_metrics=discard_undelivered_metrics,
        )


__all__ = ["ModelAdministration"]
