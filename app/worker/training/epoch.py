from __future__ import annotations

import math
from dataclasses import dataclass

DIRECT_LOSS_FIELDS = tuple(f"loss_l{index}" for index in range(6))


@dataclass
class TrainingEpochResult:
    """ML result of one completed global epoch.

    These values participate in checkpoint selection and therefore belong to
    training state. Runtime observations are deliberately represented by a
    separate telemetry type and must not affect optimization or selection.
    """

    rows: int = 0
    batches: int = 0
    loss: float = 0.0
    loss_l0: float = 0.0
    loss_l1: float = 0.0
    loss_l2: float = 0.0
    loss_l3: float = 0.0
    loss_l4: float = 0.0
    loss_l5: float = 0.0
    loss_nll: float = 0.0
    loss_ev: float = 0.0
    selection_score: float | None = None
    step: int = 0
    lr: float = 0.0
    loss_stage: int = 0
    minimum_loss_stage: int = 0
    maximum_loss_stage: int = 0

    def update(self, rows: int, loss_parts: dict[str, float]) -> None:
        total_rows = self.rows + rows
        if total_rows <= 0:
            return

        def average(current: float, value: float) -> float:
            return math.fsum((current * self.rows, value * rows)) / total_rows

        for name in ("loss", *DIRECT_LOSS_FIELDS, "loss_nll", "loss_ev"):
            setattr(self, name, average(getattr(self, name), loss_parts[name]))

        self.rows = total_rows
        self.batches += 1
        self.step = int(loss_parts.get("step", self.step))
        observed_stage = int(loss_parts.get("loss_stage", self.loss_stage))
        self.loss_stage = observed_stage
        if self.minimum_loss_stage == 0:
            self.minimum_loss_stage = observed_stage
        else:
            self.minimum_loss_stage = min(
                self.minimum_loss_stage,
                observed_stage,
            )
        self.maximum_loss_stage = max(self.maximum_loss_stage, observed_stage)

    def direct_losses(self) -> tuple[float, ...]:
        if self.rows <= 0:
            raise ValueError("selection score is incomplete: epoch has no rows")
        values = tuple(getattr(self, name) for name in DIRECT_LOSS_FIELDS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("selection score contains a non-finite component")
        return values

    def __float__(self) -> float:
        return self.loss

    def __format__(self, format_spec: str) -> str:
        return format(self.loss, format_spec)

    def __gt__(self, other: float | int) -> bool:
        return self.loss > float(other)


__all__ = ["DIRECT_LOSS_FIELDS", "TrainingEpochResult"]
