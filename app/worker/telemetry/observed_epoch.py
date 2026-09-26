from __future__ import annotations

from dataclasses import dataclass

from app.worker.telemetry.epoch_observations import EpochTelemetry
from app.worker.training.epoch import TrainingEpochResult


@dataclass
class ObservedTrainingEpoch(TrainingEpochResult):
    """Основной результат эпохи с необязательными наблюдениями по возможности."""

    telemetry: EpochTelemetry | None = None


__all__ = ["ObservedTrainingEpoch"]
