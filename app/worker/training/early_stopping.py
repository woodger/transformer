import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    improved: bool
    should_stop: bool


@dataclass
class SelectionState:
    min_delta: float
    patience: int
    active: bool = False
    best_score: float = float("inf")
    wait: int = 0

    def begin(self) -> None:
        self.active = True
        self.best_score = float("inf")
        self.wait = 0

    def update(self, score: float) -> SelectionDecision:
        if not self.active:
            raise ValueError("checkpoint selection has not started")
        if not math.isfinite(score):
            raise ValueError("checkpoint selection score must be finite")

        improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
        should_stop = self.patience > 0 and self.wait >= self.patience
        return SelectionDecision(
            improved=improved,
            should_stop=should_stop,
        )


__all__ = ["SelectionDecision", "SelectionState"]
