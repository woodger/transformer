from __future__ import annotations

from dataclasses import dataclass

from app.service.application.ports.models import ModelAdministrationStore
from app.service.application.ports.telemetry import ModelTelemetryQuery
from app.service.domain.records import ModelLifecycleRecord


@dataclass(frozen=True, slots=True)
class ModelAdministrationRecord:
    model: ModelLifecycleRecord
    metrics_delivery_status: str | None


class ModelAdministration:
    """Administrative use cases for published model generations."""

    def __init__(
        self,
        store: ModelAdministrationStore,
        telemetry: ModelTelemetryQuery | None = None,
    ) -> None:
        self.store = store
        self.telemetry = telemetry

    def list(self) -> list[ModelAdministrationRecord]:
        models = self.store.list_models()
        statuses = (
            {}
            if self.telemetry is None
            else self.telemetry.delivery_statuses(
                [model.model_ref for model in models]
            )
        )
        return [
            ModelAdministrationRecord(
                model=model,
                metrics_delivery_status=statuses.get(model.model_ref),
            )
            for model in models
        ]

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


__all__ = ["ModelAdministration", "ModelAdministrationRecord"]
