from contextlib import nullcontext
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (
    CONTEXT_MODE,
    GRAD_CLIP_NORM,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    STAGE_SIZE,
    WEIGHT_DECAY,
)
from context import context_token_ratios
from losses import (
    LOSS_STAGES,
    combined_loss,
    resolve_loss_stage,
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)
from metrics import TrainMetrics, append_metrics_jsonl
from utils import save_model, load_model, tree_stats


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
        per_week: int | None = None,
        weight_decay: float = WEIGHT_DECAY,
        metrics_path: str | None = None,
        context_mode: str = CONTEXT_MODE,
    ):
        if per_week is not None:
            stage_size = per_week

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_stage = validate_loss_stage(loss_stage)
        self.loss_schedule = validate_loss_schedule(loss_schedule)
        self.stage_size = validate_stage_size(stage_size)
        self.per_week = self.stage_size
        self.metrics_path = metrics_path
        self.context_mode = context_mode
        self.train_step = 0

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

        print(f"AMP enabled: {self.use_amp}")

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        else:
            return nullcontext()

    def _train_loader(self, loader, epoch: int) -> TrainMetrics:
        self.model.train()
        metrics = TrainMetrics(
            lr=self.optimizer.param_groups[0]["lr"],
            step=self.train_step,
            loss_stage=self._loss_stage_for(epoch),
        )
        started = time.perf_counter()

        for xb_cpu, yb_cpu in loader:
            loss_stage = self._loss_stage_for(epoch)
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
            self.train_step += 1
            loss_parts["step"] = self.train_step

            metrics.update(
                rows=batch_rows,
                loss_parts=loss_parts,
                grad_norm=float(grad_norm.detach().cpu()),
                nan_ratio=nan_ratio,
                **token_ratios,
            )

        metrics.elapsed_ms = (time.perf_counter() - started) * 1000
        return metrics

    def _loss_stage_for(self, epoch: int) -> int:
        if self.loss_schedule == "step":
            progress = self.train_step
        else:
            progress = epoch

        return resolve_loss_stage(
            progress,
            loss_schedule=self.loss_schedule,
            stage_size=self.stage_size,
            max_stage=self.loss_stage,
        )

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

        return self._train_loader(loader, epoch)

    def fit_epochs(self, X: torch.Tensor, Y: torch.Tensor, on_epoch=None):
        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        metrics_rows = []
        best_loss = float("inf")
        wait = 0
        current_loss_stage = None

        for epoch in range(self.epochs):
            metrics = self._train_loader(loader, epoch)
            metrics_rows.append(metrics)

            if on_epoch is not None:
                on_epoch(epoch, metrics)

            if metrics.loss_stage != current_loss_stage:
                current_loss_stage = metrics.loss_stage
                best_loss = float("inf")
                wait = 0

            if metrics.loss < best_loss:
                best_loss = metrics.loss
                wait = 0
            else:
                wait += 1

            if (
                self.patience > 0
                and metrics.loss_stage >= min(self.loss_stage, LOSS_STAGES)
                and wait >= self.patience
            ):
                break

        return metrics_rows

    def fit(self, X: torch.Tensor, Y: torch.Tensor, model_name: str):
        def on_epoch(epoch: int, metrics: TrainMetrics):
            stats = tree_stats(self.model.parameters())

            print(metrics.log_line(epoch=epoch + 1, norm=f"{stats['norm']:.0f}"))
            self.record_metrics(
                metrics,
                mode="fit",
                epoch=epoch + 1,
                norm=stats["norm"],
            )

        self.fit_epochs(X, Y, on_epoch=on_epoch)

        save_model(model_name, self.model)
        print("Model saved")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            return self.model(X.to(self.device))

    def load(self, model_name: str):
        load_model(model_name, self.model, self.device)

    def save(self, model_name: str):
        save_model(model_name, self.model)

    def record_metrics(self, metrics: TrainMetrics, **extra):
        extra.setdefault("context_mode", self.context_mode)
        append_metrics_jsonl(self.metrics_path, metrics, **extra)
