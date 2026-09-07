from __future__ import annotations

from typing import Protocol

from app.service.application.messages.training_telemetry import (
    ProjectedTrainingTelemetry,
)
from app.service.domain.json_types import JsonObject


class TrainingTelemetryBackendUnavailable(RuntimeError):
    pass


class TrainingTelemetryIntegrityError(RuntimeError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


class TrainingTelemetryStoredMetadataError(RuntimeError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


class TrainingTelemetrySource(Protocol):
    def load_report(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
    ) -> ProjectedTrainingTelemetry: ...

    def load_gradient_points(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
        epoch: int,
    ) -> tuple[JsonObject, ...]: ...


__all__ = [
    "TrainingTelemetryBackendUnavailable",
    "TrainingTelemetryIntegrityError",
    "TrainingTelemetrySource",
    "TrainingTelemetryStoredMetadataError",
]
