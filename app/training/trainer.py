from contextlib import nullcontext
import math
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import (
    CONTEXT_MODE,
    GRAD_CLIP_NORM,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    SAVE_BEST_CHECKPOINT,
    STAGE_SIZE,
    TRAIN_MONITOR,
    TRAIN_MONITOR_MIN_IMPROVEMENT,
    WEIGHT_DECAY,
)
from app.model.context import context_token_ratios
from app.training.early_stopping import EarlyStopping
from app.training.losses import (
    LOSS_STAGES,
    combined_loss,
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)
from app.training.loss_scheduler import LossScheduler
from app.metrics import TrainMetrics, append_metrics_jsonl
from app.training.training_state import TrainingState
from app.utils import save_model, load_model, tree_stats


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        lr: float,
        batch_size: int,
        epochs: int,
        patience: int,
        use_amp: bool = False,
        loss_stage: int = LOSS_STAGE,
        loss_schedule: str = LOSS_SCHEDULE,
        stage_size: int = STAGE_SIZE,
        weight_decay: float = WEIGHT_DECAY,
        monitor: str = TRAIN_MONITOR,
        monitor_min_improvement: float = TRAIN_MONITOR_MIN_IMPROVEMENT,
        save_best_checkpoint: bool = SAVE_BEST_CHECKPOINT,
        metrics_path: str | None = None,
        context_mode: str = CONTEXT_MODE,
        metrics_context: dict | None = None,
        model_config=None,
        train_config=None,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_stage = validate_loss_stage(loss_stage)
        self.loss_schedule = validate_loss_schedule(loss_schedule)
        self.stage_size = validate_stage_size(stage_size)
        self.monitor = self._validate_monitor(monitor)
        self.monitor_min_improvement = float(monitor_min_improvement)
        self.save_best_checkpoint = bool(save_best_checkpoint)
        self.metrics_path = metrics_path
        self.context_mode = context_mode
        self.model_config = model_config
        self.train_config = train_config
        self.best_monitor = float("inf")
        self.best_state_dict = None
        self.best_metrics = None
        self.best_frame = None
        self.best_epoch = None
        self.state = TrainingState()
        self.loss_scheduler = LossScheduler(
            loss_schedule=self.loss_schedule,
            stage_size=self.stage_size,
            max_stage=self.loss_stage,
        )
        self.metrics_context = {
            "batch_size": self.batch_size,
            "loss_schedule": self.loss_schedule,
            "stage_size": self.stage_size,
            "max_loss_stage": self.loss_stage,
            "device": str(self.device),
            "monitor": self.monitor,
            "monitor_min_improvement": self.monitor_min_improvement,
        }
        if metrics_context:
            self.metrics_context.update(metrics_context)

        # AMP включаем только если GPU и user просил
        self.use_amp = bool(use_amp and device.type == "cuda")

        # GradScaler для AMP
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if use_amp and device.type != "cuda":
            print("AMP requested but CUDA not available — disabled")

    def _validate_monitor(self, monitor: str) -> str:
        choices = ("loss", "ret_mae", "ret_mae_skill")
        if monitor not in choices:
            raise ValueError(f"monitor must be one of: {', '.join(choices)}")
        return monitor

    @property
    def train_step(self) -> int:
        return self.state.train_step

    @train_step.setter
    def train_step(self, value: int):
        self.state.train_step = value

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        else:
            return nullcontext()

    def _train_loader(self, loader) -> TrainMetrics:
        self.model.train()
        metrics = TrainMetrics(
            lr=self.optimizer.param_groups[0]["lr"],
            step=self.state.train_step,
            loss_stage=self._loss_stage_for(),
        )
        started = time.perf_counter()

        for xb_cpu, yb_cpu in loader:
            loss_stage = self._loss_stage_for()
            batch_rows = xb_cpu.size(0)
            nan_ratio = float(torch.isnan(xb_cpu).float().mean())
            token_ratios = context_token_ratios(xb_cpu, self.context_mode)

            xb = xb_cpu.to(self.device)
            yb = yb_cpu.to(self.device)

            self.optimizer.zero_grad()

            with self._autocast():
                preds = self.model(xb)
                loss, loss_parts = combined_loss(
                    preds,
                    yb,
                    loss_stage,
                    return_parts=True,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), GRAD_CLIP_NORM
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_parts["step"] = self.state.finish_step()

            metrics.update(
                rows=batch_rows,
                loss_parts=loss_parts,
                grad_norm=float(grad_norm.detach().cpu()),
                nan_ratio=nan_ratio,
                **token_ratios,
            )

        metrics.elapsed_ms = (time.perf_counter() - started) * 1000
        return metrics

    def _monitor_value(self, metrics: TrainMetrics) -> float:
        if self.monitor == "loss":
            return metrics.loss
        if self.monitor == "ret_mae":
            return metrics.ret_mae
        return metrics.ret_mae_skill

    def _baseline_passed(self, metrics: TrainMetrics) -> bool:
        if self.monitor == "loss":
            return True
        if metrics.ret_mae_baseline <= 0.0:
            return False

        threshold = metrics.ret_mae_baseline * (1.0 - self.monitor_min_improvement)
        return metrics.ret_mae < threshold

    def _observe_metrics(
        self,
        metrics: TrainMetrics,
        frame: int | None = None,
        epoch: int | None = None,
    ) -> dict:
        monitor_value = self._monitor_value(metrics)
        baseline_passed = self._baseline_passed(metrics)
        checkpoint_best = False

        if (
            self.save_best_checkpoint
            and baseline_passed
            and math.isfinite(monitor_value)
            and monitor_value < self.best_monitor
        ):
            self.best_monitor = monitor_value
            self.best_state_dict = self._snapshot_state_dict()
            self.best_metrics = metrics.to_dict()
            self.best_frame = frame
            self.best_epoch = epoch
            checkpoint_best = True

        payload = {
            "monitor_value": monitor_value,
            "baseline_passed": baseline_passed,
            "checkpoint_best": checkpoint_best,
        }
        if math.isfinite(self.best_monitor):
            payload["best_monitor"] = self.best_monitor
        else:
            payload["best_monitor"] = None

        return payload

    def _snapshot_state_dict(self) -> dict:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _restore_best_state_dict(self):
        if self.best_state_dict is None:
            return

        state_dict = {
            key: value.to(self.device)
            for key, value in self.best_state_dict.items()
        }
        self.model.load_state_dict(state_dict)

    def _loss_stage_for(self) -> int:
        return self.loss_scheduler.stage_for(self.state)

    def fit_batch(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        epoch: int = 0,
    ) -> TrainMetrics:
        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.state.begin_epoch(epoch)
        return self._train_loader(loader)

    def fit_epochs(self, X: torch.Tensor, Y: torch.Tensor, on_epoch=None, frame: int | None = None):
        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        metrics_rows = []
        stopper = EarlyStopping(
            patience=self.patience,
            min_stage=min(self.loss_stage, LOSS_STAGES),
        )
        self.state.begin_frame(frame)

        for epoch in range(self.epochs):
            self.state.begin_epoch(epoch)
            metrics = self._train_loader(loader)
            metrics_rows.append(metrics)
            monitor_payload = self._observe_metrics(
                metrics,
                frame=frame,
                epoch=epoch + 1,
            )

            if on_epoch is not None:
                on_epoch(epoch, metrics, monitor_payload)

            should_stop = stopper.update(
                monitor_payload["monitor_value"],
                metrics.loss_stage,
            )
            self.state.finish_epoch()
            if should_stop:
                break

        return metrics_rows

    def fit(self, X: torch.Tensor, Y: torch.Tensor, model_name: str):
        def on_epoch(epoch: int, metrics: TrainMetrics, monitor_payload: dict):
            stats = tree_stats(self.model.parameters())

            print(metrics.console_line(
                epoch=epoch + 1,
                norm=f"{stats['norm']:.0f}",
                **self.metrics_context,
                **monitor_payload,
            ))
            self.record_metrics(
                metrics,
                mode="fit",
                epoch=epoch + 1,
                norm=stats["norm"],
                **monitor_payload,
            )

        print(self.config_line())
        self.fit_epochs(X, Y, on_epoch=on_epoch)

        self.save(model_name)
        print("Model saved")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            return self.model(X.to(self.device))

    def load(self, model_name: str):
        load_model(model_name, self.model, self.device)

    def save(self, model_name: str):
        if self.save_best_checkpoint and self.best_state_dict is not None:
            self._restore_best_state_dict()

        save_model(
            model_name,
            self.model,
            model_config=self.model_config,
            train_config=self.train_config,
            extra={
                "checkpoint_selection": {
                    "monitor": self.monitor,
                    "monitor_min_improvement": self.monitor_min_improvement,
                    "best_monitor": (
                        self.best_monitor if math.isfinite(self.best_monitor) else None
                    ),
                    "best_frame": self.best_frame,
                    "best_epoch": self.best_epoch,
                    "baseline_passed": self.best_state_dict is not None,
                    "source": (
                        "best_monitor"
                        if self.best_state_dict is not None
                        else "current"
                    ),
                },
            },
        )

    def config_line(self) -> str:
        context = self.metrics_context
        fields = {
            "device": context["device"],
            "batch_size": context["batch_size"],
            "lr": f"{self.optimizer.param_groups[0]['lr']:.6g}",
        }
        for key in ("hidden", "layers", "seq_len"):
            if key in context:
                fields[key] = context[key]
        fields.update({
            "loss_schedule": context["loss_schedule"],
            "stage_size": context["stage_size"],
            "max_loss_stage": context["max_loss_stage"],
            "monitor": context["monitor"],
            "monitor_min_improvement": f"{context['monitor_min_improvement']:.6g}",
            "context_mode": self.context_mode,
            "amp": self.use_amp,
        })
        return "config " + " ".join(
            f"{key}={value}" for key, value in fields.items()
        )

    def record_metrics(self, metrics: TrainMetrics, **extra):
        payload = {**self.metrics_context, **extra}
        payload.setdefault("context_mode", self.context_mode)
        append_metrics_jsonl(self.metrics_path, metrics, **payload)
