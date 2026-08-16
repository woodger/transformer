from dataclasses import dataclass

from app.contracts.worker.v4.config import (
    DEFAULT_LOSS_SCHEDULE,
    DEFAULT_LOSS_STAGE,
    DEFAULT_STAGE_SIZE,
)
from app.worker.training.losses import (
    resolve_loss_stage,
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)
from app.worker.training.training_state import TrainingState


@dataclass(frozen=True)
class LossScheduler:
    loss_schedule: str = DEFAULT_LOSS_SCHEDULE
    stage_size: int = DEFAULT_STAGE_SIZE
    max_stage: int = DEFAULT_LOSS_STAGE

    def __post_init__(self) -> None:
        validate_loss_schedule(self.loss_schedule)
        validate_stage_size(self.stage_size)
        validate_loss_stage(self.max_stage)

    def stage_for(self, state: TrainingState) -> int:
        if self.loss_schedule == "step":
            progress = state.train_step
        elif self.loss_schedule == "epoch":
            progress = state.frame_epoch
        else:
            progress = 0

        return resolve_loss_stage(
            progress,
            loss_schedule=self.loss_schedule,
            stage_size=self.stage_size,
            max_stage=self.max_stage,
        )
