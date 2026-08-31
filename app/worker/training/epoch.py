from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.contracts.ml import TARGET_IDENTITIES, canonical_targets
from app.worker.training.losses import MaterializedLossStatistics


def _empty_loss_values() -> dict[str, float]:
    return {}


@dataclass
class TrainingEpochResult:
    """Core ML result of one completed global epoch."""

    targets: tuple[str, ...] = TARGET_IDENTITIES
    auxiliary_operators: tuple[str, ...] = ()
    rows: int = 0
    batches: int = 0
    loss: float = 0.0
    direct_loss_values: dict[str, float] = field(
        default_factory=_empty_loss_values
    )
    auxiliary_loss_values: dict[str, float] = field(
        default_factory=_empty_loss_values
    )
    selection_score: float | None = None
    step: int = 0
    lr: float = 0.0

    def __post_init__(self) -> None:
        self.targets = canonical_targets(self.targets)
        self.direct_loss_values = {
            target: float(self.direct_loss_values.get(target, 0.0))
            for target in self.targets
        }
        self.auxiliary_loss_values = {
            operator: float(self.auxiliary_loss_values.get(operator, 0.0))
            for operator in self.auxiliary_operators
        }

    def update(
        self,
        rows: int,
        statistics: MaterializedLossStatistics,
        *,
        step: int,
    ) -> None:
        total_rows = self.rows + rows
        if total_rows <= 0:
            return
        if len(statistics.direct_losses) != len(self.targets):
            raise ValueError("direct loss statistics differ from target selection")
        if tuple(
            operator for operator, _value in statistics.auxiliary_losses
        ) != self.auxiliary_operators:
            raise ValueError("auxiliary loss statistics differ from objective")

        def average(current: float, value: float) -> float:
            return math.fsum((current * self.rows, value * rows)) / total_rows

        self.loss = average(self.loss, statistics.loss)
        for target, value in zip(
            self.targets,
            statistics.direct_losses,
            strict=True,
        ):
            self.direct_loss_values[target] = average(
                self.direct_loss_values[target],
                value,
            )
        for operator, value in statistics.auxiliary_losses:
            self.auxiliary_loss_values[operator] = average(
                self.auxiliary_loss_values[operator],
                value,
            )

        self.rows = total_rows
        self.batches += 1
        self.step = step

    def direct_losses(self) -> tuple[float, ...]:
        if self.rows <= 0:
            raise ValueError("selection score is incomplete: epoch has no rows")
        values = tuple(self.direct_loss_values[target] for target in self.targets)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("selection score contains a non-finite component")
        return values

    def __float__(self) -> float:
        return self.loss

    def __format__(self, format_spec: str) -> str:
        return format(self.loss, format_spec)

    def __gt__(self, other: float | int) -> bool:
        return self.loss > float(other)


__all__ = ["TrainingEpochResult"]
