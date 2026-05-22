from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_stage: int

    best_loss: float = float("inf")
    wait: int = 0
    current_stage: int | None = None

    def update(self, loss: float, stage: int) -> bool:
        if stage != self.current_stage:
            self.current_stage = stage
            self.best_loss = float("inf")
            self.wait = 0

        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1

        return self.patience > 0 and stage >= self.min_stage and self.wait >= self.patience
