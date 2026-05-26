from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    patience: int
    min_stage: int

    best_score: float = float("inf")
    wait: int = 0
    current_stage: int | None = None

    def update(self, score: float, stage: int) -> bool:
        if stage != self.current_stage:
            self.current_stage = stage
            self.best_score = float("inf")
            self.wait = 0

        if math.isfinite(score) and score < self.best_score:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1

        return self.patience > 0 and stage >= self.min_stage and self.wait >= self.patience
