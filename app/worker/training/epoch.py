from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.worker.training.losses import MaterializedLossStatistics


def _empty_loss_values() -> dict[str, float]:
    return {}


@dataclass
class TrainingEpochResult:
    """Основной ML-результат одной завершённой глобальной эпохи."""

    targets: tuple[str, ...]
    direct_components: tuple[tuple[str, str], ...]
    auxiliary_components: tuple[tuple[str, str], ...] = ()
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
        if not self.targets or len(self.targets) != len(set(self.targets)):
            raise ValueError("epoch target identities must be non-empty and unique")
        if len(self.direct_components) != len(self.targets):
            raise ValueError("epoch direct components differ from target layout")
        self.direct_loss_values = {
            identity: float(self.direct_loss_values.get(identity, 0.0))
            for identity, _operator in self.direct_components
        }
        self.auxiliary_loss_values = {
            identity: float(self.auxiliary_loss_values.get(identity, 0.0))
            for identity, _operator in self.auxiliary_components
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
            (identity, operator)
            for identity, operator, _value in statistics.auxiliary_losses
        ) != self.auxiliary_components:
            raise ValueError("auxiliary loss statistics differ from objective")

        def average(current: float, value: float) -> float:
            return math.fsum((current * self.rows, value * rows)) / total_rows

        self.loss = average(self.loss, statistics.loss)
        for (identity, _operator), value in zip(
            self.direct_components,
            statistics.direct_losses,
            strict=True,
        ):
            self.direct_loss_values[identity] = average(
                self.direct_loss_values[identity],
                value,
            )
        for identity, _operator, value in statistics.auxiliary_losses:
            self.auxiliary_loss_values[identity] = average(
                self.auxiliary_loss_values[identity],
                value,
            )

        self.rows = total_rows
        self.batches += 1
        self.step = step

    def direct_losses(self) -> tuple[float, ...]:
        if self.rows <= 0:
            raise ValueError("selection score is incomplete: epoch has no rows")
        values = tuple(
            self.direct_loss_values[identity]
            for identity, _operator in self.direct_components
        )
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
