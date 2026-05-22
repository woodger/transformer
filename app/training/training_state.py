from dataclasses import dataclass


@dataclass
class TrainingState:
    frame: int = 0
    frame_epoch: int = 0
    global_epoch: int = 0
    train_step: int = 0

    def begin_frame(self, frame: int | None = None):
        if frame is not None:
            self.frame = frame
        self.frame_epoch = 0

    def begin_epoch(self, frame_epoch: int):
        self.frame_epoch = frame_epoch

    def finish_epoch(self):
        self.global_epoch += 1

    def finish_step(self):
        self.train_step += 1
        return self.train_step
