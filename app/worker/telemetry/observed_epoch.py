from __future__ import annotations

from dataclasses import dataclass

from app.worker.telemetry.epoch_observations import EpochTelemetry
from app.worker.training.epoch import TrainingEpochResult


@dataclass
class ObservedTrainingEpoch(TrainingEpochResult):
    """Core epoch result accompanied by optional best-effort observations."""

    telemetry: EpochTelemetry | None = None


__all__ = ["ObservedTrainingEpoch"]
