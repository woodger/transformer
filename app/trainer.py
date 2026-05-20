import torch
import time
from torch.utils.data import DataLoader, TensorDataset
from contextlib import nullcontext

from context import context_valid_token_ratio
from losses import combined_loss
from metrics import TrainMetrics, append_metrics_jsonl
from utils import save_model, load_model, tree_stats
from config import WEIGHT_DECAY, GRAD_CLIP_NORM, PER_WEEK


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
        per_week: int = PER_WEEK,
        weight_decay: float = WEIGHT_DECAY,
        metrics_path: str | None = None,
    ):
        if per_week <= 0:
            raise ValueError("per_week must be a positive integer")

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.per_week = per_week
        self.metrics_path = metrics_path

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
            week=epoch // self.per_week + 1,
        )
        started = time.perf_counter()

        for xb_cpu, yb_cpu in loader:
            batch_rows = xb_cpu.size(0)
            nan_ratio = float(torch.isnan(xb_cpu).float().mean())
            valid_token_ratio = context_valid_token_ratio(xb_cpu)

            xb = xb_cpu.to(self.device)
            yb = yb_cpu.to(self.device)

            self.optimizer.zero_grad()

            with self._autocast():
                preds = self.model(xb)
                loss, loss_parts = combined_loss(
                    preds,
                    yb,
                    epoch,
                    self.per_week,
                    return_parts=True,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), GRAD_CLIP_NORM
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics.update(
                rows=batch_rows,
                loss_parts=loss_parts,
                grad_norm=float(grad_norm.detach().cpu()),
                nan_ratio=nan_ratio,
                valid_token_ratio=valid_token_ratio,
            )

        metrics.elapsed_ms = (time.perf_counter() - started) * 1000
        return metrics

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

    def fit(self, X: torch.Tensor, Y: torch.Tensor, model_name: str):
        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        best_loss = float("inf")
        wait = 0

        for epoch in range(self.epochs):
            metrics = self._train_loader(loader, epoch)
            stats = tree_stats(self.model.parameters())

            print(metrics.log_line(epoch=epoch + 1, norm=f"{stats['norm']:.0f}"))
            self.record_metrics(
                metrics,
                mode="fit",
                epoch=epoch + 1,
                norm=stats["norm"],
            )

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
        append_metrics_jsonl(self.metrics_path, metrics, **extra)
