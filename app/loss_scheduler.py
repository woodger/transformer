from dataclasses import dataclass

from config import LOSS_SCHEDULE, LOSS_STAGE, STAGE_SIZE
from losses import resolve_loss_stage, validate_loss_schedule, validate_loss_stage, validate_stage_size
from training_state import TrainingState


@dataclass(frozen=True)
class LossScheduler:
    loss_schedule: str = LOSS_SCHEDULE
    stage_size: int = STAGE_SIZE
    max_stage: int = LOSS_STAGE

    def __post_init__(self):
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
