import copy
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import (
    CONTEXT_MODE,
    GRAD_CLIP_NORM,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    SAVE_BEST_CHECKPOINT,
    SEED,
    STAGE_SIZE,
    TRAIN_MONITOR,
    TRAIN_MONITOR_MIN_IMPROVEMENT,
    WEIGHT_DECAY,
)
from app.worker.metrics import TrainMetrics, append_metrics_jsonl
from app.worker.model.context import context_missingness_ratios
from app.worker.training.early_stopping import EarlyStopping
from app.worker.training.loss_scheduler import LossScheduler
from app.worker.training.losses import (
    LOSS_STAGES,
    combined_loss,
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)
from app.worker.training.training_state import TrainingState
from app.worker.utils import load_model, save_model, tree_stats

_MAX_SHUFFLE_WINDOW_BATCHES = 32
_MAX_SHUFFLE_WINDOW_BYTES = 64 * 1024 * 1024


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
        seed: int = SEED,
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
        self.seed = seed
        self.best_monitor = float("inf")
        self.best_state_dict = None
        self.best_metrics = None
        self.best_frame = None
        self.best_epoch = None
        self.state = TrainingState()
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_stage=min(self.loss_stage, LOSS_STAGES),
        )
        self.training_complete = False
        self._payload_shuffle_generator = torch.Generator()
        self._payload_shuffle_generator.manual_seed(self.seed)
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

        self.use_amp = bool(use_amp and device.type == "cuda")
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

    def _train_loaders(self, loaders) -> TrainMetrics:
        self.model.train()
        metrics = TrainMetrics(
            lr=self.optimizer.param_groups[0]["lr"],
            step=self.state.train_step,
            loss_stage=self._loss_stage_for(),
        )
        started = time.perf_counter()

        # Phase timers stay host-side and must never force CUDA synchronization.
        for loader in loaders:
            batches = iter(loader)
            while True:
                phase_started = time.perf_counter()
                try:
                    xb_cpu, yb_cpu = next(batches)
                except StopIteration:
                    break
                metrics.input_pipeline_ms += (
                    time.perf_counter() - phase_started
                ) * 1000

                loss_stage = self._loss_stage_for()
                batch_rows = xb_cpu.size(0)
                phase_started = time.perf_counter()
                missingness_ratios = context_missingness_ratios(
                    xb_cpu,
                    self.context_mode,
                )
                metrics.missing_stats_ms += (
                    time.perf_counter() - phase_started
                ) * 1000

                phase_started = time.perf_counter()
                xb = xb_cpu.to(self.device)
                yb = yb_cpu.to(self.device)
                metrics.host_to_device_ms += (
                    time.perf_counter() - phase_started
                ) * 1000

                phase_started = time.perf_counter()
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
                    **missingness_ratios,
                )
                metrics.train_step_ms += (
                    time.perf_counter() - phase_started
                ) * 1000
            del loader

        metrics.elapsed_ms = (time.perf_counter() - started) * 1000
        return metrics

    def _train_loader(self, loader) -> TrainMetrics:
        return self._train_loaders((loader,))

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
        loader = self._data_loader(X, Y)

        self.state.begin_epoch(epoch)
        return self._train_loader(loader)

    def fit_epochs(self, X: torch.Tensor, Y: torch.Tensor, on_epoch=None, frame: int | None = None):
        loader = self._data_loader(X, Y)
        return self._fit_loader_epochs(
            lambda: (loader,),
            on_epoch=on_epoch,
            frame=frame,
        )

    def fit_payloads(self, payloads, on_epoch=None):
        """Train global epochs over a payload-independent row stream.

        ``payloads`` is a callable so durable inputs can be reopened for every
        epoch. Optimizer batches and bounded shuffle windows may cross payload
        boundaries, so transport partitioning cannot change the trajectory.
        """
        def loaders():
            yield self._payload_batches(
                payloads(),
                self._payload_shuffle_generator,
            )

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
            stopper=self.early_stopping,
        )

    def _payload_batches(self, payloads, generator):
        window_rows = None
        source_buffer = None
        target_buffer = None
        buffered_rows = 0

        for source, targets in payloads:
            payload_rows = source.size(0)
            if targets.size(0) != payload_rows:
                raise ValueError("source and target row counts must match")
            if payload_rows == 0:
                continue

            if source_buffer is None:
                row_bytes = (
                    source[0].numel() * source.element_size()
                    + targets[0].numel() * targets.element_size()
                )
                batch_bytes = self.batch_size * row_bytes
                window_batches = min(
                    _MAX_SHUFFLE_WINDOW_BATCHES,
                    max(1, _MAX_SHUFFLE_WINDOW_BYTES // batch_bytes),
                )
                window_rows = self.batch_size * window_batches
                source_buffer = torch.empty(
                    (window_rows, *source.shape[1:]),
                    dtype=source.dtype,
                    device=source.device,
                )
                target_buffer = torch.empty(
                    (window_rows, *targets.shape[1:]),
                    dtype=targets.dtype,
                    device=targets.device,
                )

            offset = 0
            while offset < payload_rows:
                copied_rows = min(
                    window_rows - buffered_rows,
                    payload_rows - offset,
                )
                buffer_end = buffered_rows + copied_rows
                payload_end = offset + copied_rows
                source_buffer[buffered_rows:buffer_end].copy_(
                    source[offset:payload_end]
                )
                target_buffer[buffered_rows:buffer_end].copy_(
                    targets[offset:payload_end]
                )
                buffered_rows = buffer_end
                offset = payload_end

                if buffered_rows == window_rows:
                    yield from self._shuffled_batches(
                        source_buffer,
                        target_buffer,
                        buffered_rows,
                        generator,
                    )
                    buffered_rows = 0

            del source, targets

        if buffered_rows:
            yield from self._shuffled_batches(
                source_buffer,
                target_buffer,
                buffered_rows,
                generator,
            )

    def _shuffled_batches(self, source, targets, rows: int, generator):
        order = torch.randperm(
            rows,
            generator=generator,
            device=source.device,
        )
        for offset in range(0, rows, self.batch_size):
            indices = order[offset:offset + self.batch_size]
            yield (
                source.index_select(0, indices),
                targets.index_select(0, indices),
            )

    def _data_loader(self, X: torch.Tensor, Y: torch.Tensor):
        dataset = TensorDataset(X, Y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def _fit_loader_epochs(
        self,
        loaders,
        on_epoch=None,
        frame: int | None = None,
        *,
        start_epoch: int = 0,
        stopper: EarlyStopping | None = None,
        on_epoch_committed=None,
    ):
        metrics_rows = []
        stopper = stopper or EarlyStopping(
            patience=self.patience,
            min_stage=min(self.loss_stage, LOSS_STAGES),
        )
        self.state.begin_frame(frame)
        self.training_complete = False

        for epoch in range(start_epoch, self.epochs):
            self.state.begin_epoch(epoch)
            metrics = self._train_loaders(loaders())
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
            self.training_complete = (
                should_stop or self.state.global_epoch >= self.epochs
            )
            if on_epoch_committed is not None:
                on_epoch_committed(
                    epoch,
                    metrics,
                    monitor_payload,
                    self.training_complete,
                )
            if should_stop:
                break

        return metrics_rows

    def fit_payloads_resumable(
        self,
        payloads,
        *,
        on_epoch=None,
        on_epoch_committed=None,
    ):
        """Train job-wide epochs and expose only complete recovery boundaries."""

        def loaders():
            yield self._payload_batches(
                payloads(),
                self._payload_shuffle_generator,
            )

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
            stopper=self.early_stopping,
            on_epoch_committed=on_epoch_committed,
        )

    def fit_streaming_payloads(
        self,
        first_epoch_payloads,
        closed_payloads,
        *,
        on_epoch=None,
        on_epoch_committed=None,
    ):
        """Train epoch zero from an open stream, then replay closed input.

        The first iterable may block at the durable input frontier. Its EOF is
        the explicit input-close boundary. Every later epoch reopens the same
        complete ordered dataset through ``closed_payloads``.
        """

        first_epoch = True

        def loaders():
            nonlocal first_epoch
            if first_epoch:
                payloads = first_epoch_payloads
                first_epoch = False
            else:
                payloads = closed_payloads()
            yield self._payload_batches(
                payloads,
                self._payload_shuffle_generator,
            )

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
            stopper=self.early_stopping,
            on_epoch_committed=on_epoch_committed,
        )

    def recovery_state_dict(self) -> dict:
        """Return the complete trusted state needed to resume a fit."""

        cuda_rng_state = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_rng_state = [
                state.detach().cpu()
                for state in torch.cuda.get_rng_state_all()
            ]
        return {
            "model_state_dict": _tree_to_cpu(self.model.state_dict()),
            "optimizer_state_dict": _tree_to_cpu(
                self.optimizer.state_dict()
            ),
            "scaler_state_dict": copy.deepcopy(self.scaler.state_dict()),
            "training_state": asdict(self.state),
            "early_stopping": asdict(self.early_stopping),
            "selection": {
                "best_monitor": self.best_monitor,
                "best_state_dict": (
                    None
                    if self.best_state_dict is None
                    else _tree_to_cpu(self.best_state_dict)
                ),
                "best_metrics": copy.deepcopy(self.best_metrics),
                "best_frame": self.best_frame,
                "best_epoch": self.best_epoch,
            },
            "payload_shuffle_generator_state": (
                self._payload_shuffle_generator.get_state()
            ),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": cuda_rng_state,
            },
            "training_complete": self.training_complete,
        }

    def load_recovery_state_dict(self, payload: dict) -> None:
        """Restore a state produced by :meth:`recovery_state_dict`."""

        if not isinstance(payload, dict):
            raise ValueError("training recovery state must be an object")
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
            "training_state",
            "early_stopping",
            "selection",
            "payload_shuffle_generator_state",
            "rng",
            "training_complete",
        }
        if set(payload) != required:
            raise ValueError("training recovery state has invalid fields")

        training_state = payload["training_state"]
        early_stopping = payload["early_stopping"]
        selection = payload["selection"]
        rng = payload["rng"]
        if not isinstance(training_state, dict) or set(training_state) != {
            "frame",
            "frame_epoch",
            "global_epoch",
            "train_step",
        }:
            raise ValueError("training recovery progress is invalid")
        if not isinstance(early_stopping, dict) or set(early_stopping) != {
            "patience",
            "min_stage",
            "best_score",
            "wait",
            "current_stage",
        }:
            raise ValueError("training recovery early stopping is invalid")
        if not isinstance(selection, dict) or set(selection) != {
            "best_monitor",
            "best_state_dict",
            "best_metrics",
            "best_frame",
            "best_epoch",
        }:
            raise ValueError("training recovery checkpoint selection is invalid")
        if not isinstance(rng, dict) or set(rng) != {
            "python",
            "numpy",
            "torch",
            "cuda",
        }:
            raise ValueError("training recovery random state is invalid")
        if early_stopping["patience"] != self.patience:
            raise ValueError("training recovery patience does not match")
        if early_stopping["min_stage"] != min(
            self.loss_stage,
            LOSS_STAGES,
        ):
            raise ValueError(
                "training recovery loss stage does not match"
            )

        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to(self.optimizer, self.device)
        self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.state = TrainingState(**training_state)
        self.early_stopping = EarlyStopping(**early_stopping)
        self.best_monitor = selection["best_monitor"]
        self.best_state_dict = selection["best_state_dict"]
        self.best_metrics = selection["best_metrics"]
        self.best_frame = selection["best_frame"]
        self.best_epoch = selection["best_epoch"]
        self._payload_shuffle_generator.set_state(
            payload["payload_shuffle_generator_state"]
        )
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        cuda_rng_state = rng["cuda"]
        if cuda_rng_state is not None:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError(
                    "CUDA training recovery requires an available CUDA device"
                )
            torch.cuda.set_rng_state_all(cuda_rng_state)
        self.training_complete = bool(payload["training_complete"])

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
            if X.size(0) == 0:
                return self.model(X.to(self.device))

            predictions = None
            for offset in range(0, X.size(0), self.batch_size):
                batch = X[offset:offset + self.batch_size].to(self.device)
                output = self.model(batch)
                if predictions is None:
                    predictions = torch.empty(
                        (X.size(0), *output.shape[1:]),
                        dtype=output.dtype,
                        device=output.device,
                    )
                predictions[offset:offset + output.size(0)].copy_(output)

            return predictions

    def load(self, model_name: str):
        load_model(model_name, self.model, self.device)

    def load_payload(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint["state_dict"])

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


def _tree_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _tree_to_cpu(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_to(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
