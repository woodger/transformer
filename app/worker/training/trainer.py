from __future__ import annotations

import copy
import math
import queue
import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from typing import Protocol, TypedDict, cast, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import (
    CONTEXT_MODE,
    GRAD_CLIP_NORM,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    SEED,
    STAGE_SIZE,
    WEIGHT_DECAY,
)
from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v3.config import (
    DEFAULT_DIRECT_LOSS_WEIGHTS,
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v3.objective import objective_config_sha256
from app.worker.metrics import TrainMetrics, append_metrics_jsonl
from app.worker.model.context import context_missingness_ratios
from app.worker.model.transformer import public_predictions
from app.worker.training.early_stopping import SelectionState
from app.worker.training.loss_scheduler import LossScheduler
from app.worker.training.losses import (
    combined_loss,
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)
from app.worker.training.training_state import TrainingState
from app.worker.utils import load_model, save_model, tree_stats

_MAX_SHUFFLE_WINDOW_BATCHES = 32
_MAX_SHUFFLE_WINDOW_BYTES = 64 * 1024 * 1024

TensorBatch = tuple[torch.Tensor, torch.Tensor]
Payloads = Iterable[TensorBatch]
PayloadFactory = Callable[[], Payloads]


class SelectionPayload(TypedDict):
    selection_score: float | None
    checkpoint_best: bool
    should_stop: bool
    best_selection_score: float | None


EpochCallback = Callable[[int, TrainMetrics, SelectionPayload], None]
EpochCommittedCallback = Callable[
    [int, TrainMetrics, SelectionPayload, bool],
    None,
]


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class _PrefetchEnd:
    pass


_PREFETCH_END = _PrefetchEnd()


@dataclass(frozen=True)
class _PrefetchError:
    error: BaseException


class _BatchPrefetcher:
    """Prepare at most one closed-input batch ahead of the trainer."""

    def __init__(self, batches: Iterable[TensorBatch]) -> None:
        self._batches = iter(batches)
        self._queue: queue.Queue[
            TensorBatch | _PrefetchError | _PrefetchEnd
        ] = queue.Queue(maxsize=1)
        self._slot = threading.Semaphore(1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._produce,
            name="transformer-batch-prefetch",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> _BatchPrefetcher:
        return self

    def __next__(self) -> TensorBatch:
        message = self._queue.get()
        self._slot.release()
        if isinstance(message, _PrefetchEnd):
            self._thread.join()
            raise StopIteration
        if isinstance(message, _PrefetchError):
            self._thread.join()
            raise message.error
        return message

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.1)

    def _produce(self) -> None:
        try:
            while self._reserve_slot():
                try:
                    message = next(self._batches)
                except StopIteration:
                    self._queue.put(_PREFETCH_END)
                    return
                except BaseException as exc:
                    self._queue.put(_PrefetchError(exc))
                    return
                if self._stop.is_set():
                    self._slot.release()
                    return
                self._queue.put(message)
        finally:
            if isinstance(self._batches, _Closable):
                self._batches.close()

    def _reserve_slot(self) -> bool:
        while not self._stop.is_set():
            if self._slot.acquire(timeout=0.05):
                return True
        return False


class Trainer:
    """Own optimization, checkpoint selection, and resumable training state.

    AMP is enabled only on CUDA. Recovery serialization includes model,
    optimizer, scaler, RNG, shuffle, early-stopping, and checkpoint-selection
    state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        lr: float,
        batch_size: int,
        epochs: int,
        use_amp: bool = False,
        loss_stage: int = LOSS_STAGE,
        loss_schedule: str = LOSS_SCHEDULE,
        stage_size: int = STAGE_SIZE,
        weight_decay: float = WEIGHT_DECAY,
        direct_loss_weights: tuple[float, ...] = DEFAULT_DIRECT_LOSS_WEIGHTS,
        selection: CheckpointSelectionConfig | None = None,
        metrics_path: str | None = None,
        context_mode: str = CONTEXT_MODE,
        metrics_context: Mapping[str, JsonValue] | None = None,
        model_config: ModelConfig | None = None,
        train_config: TrainConfig | None = None,
        data_contract: Mapping[str, object] | None = None,
        seed: int = SEED,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_stage = validate_loss_stage(loss_stage)
        self.loss_schedule = validate_loss_schedule(loss_schedule)
        self.stage_size = validate_stage_size(stage_size)
        self.direct_loss_weights = tuple(float(value) for value in direct_loss_weights)
        if len(self.direct_loss_weights) != 6 or any(
            not math.isfinite(value) or value <= 0
            for value in self.direct_loss_weights
        ):
            raise ValueError("direct_loss_weights must contain six positive values")
        self.selection = selection
        self.metrics_path = metrics_path
        self.context_mode = context_mode
        self.model_config = model_config
        self.train_config = train_config
        self.data_contract = (
            None if data_contract is None else dict(data_contract)
        )
        self.seed = seed
        self.best_selection_score: float = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_metrics: JsonObject | None = None
        self.best_frame: int | None = None
        self.best_epoch: int | None = None
        self.state = TrainingState()
        self.selection_state = (
            None
            if selection is None
            else SelectionState(selection.min_delta, selection.patience)
        )
        self.maximum_stage_completed = False
        self.training_complete = False
        self._payload_shuffle_generator = torch.Generator()
        self._payload_shuffle_generator.manual_seed(self.seed)
        self.loss_scheduler = LossScheduler(
            loss_schedule=self.loss_schedule,
            stage_size=self.stage_size,
            max_stage=self.loss_stage,
        )
        self.metrics_context: dict[str, JsonValue] = {
            "batch_size": self.batch_size,
            "loss_schedule": self.loss_schedule,
            "stage_size": self.stage_size,
            "max_loss_stage": self.loss_stage,
            "device": str(self.device),
            "selection_enabled": self.selection is not None,
        }
        if metrics_context:
            self.metrics_context.update(metrics_context)

        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if use_amp and device.type != "cuda":
            print("AMP requested but CUDA not available — disabled")

    @property
    def train_step(self) -> int:
        return self.state.train_step

    @train_step.setter
    def train_step(self, value: int) -> None:
        self.state.train_step = value

    def _autocast(self) -> AbstractContextManager[object]:
        if self.use_amp:
            return torch.autocast(device_type="cuda", enabled=True)
        else:
            return nullcontext()

    def _train_loaders(
        self,
        loaders: Iterable[Iterable[TensorBatch]],
    ) -> TrainMetrics:
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
            try:
                while True:
                    phase_started = time.perf_counter()
                    try:
                        batch_features_cpu, batch_targets_cpu = next(batches)
                    except StopIteration:
                        break
                    metrics.input_pipeline_ms += (
                        time.perf_counter() - phase_started
                    ) * 1000

                    loss_stage = self._loss_stage_for()
                    batch_rows = batch_features_cpu.size(0)
                    phase_started = time.perf_counter()
                    missingness_ratios = context_missingness_ratios(
                        batch_features_cpu,
                        self.context_mode,
                    )
                    metrics.missing_stats_ms += (
                        time.perf_counter() - phase_started
                    ) * 1000

                    phase_started = time.perf_counter()
                    batch_features = batch_features_cpu.to(self.device)
                    batch_targets = batch_targets_cpu.to(self.device)
                    metrics.host_to_device_ms += (
                        time.perf_counter() - phase_started
                    ) * 1000

                    phase_started = time.perf_counter()
                    self.optimizer.zero_grad()

                    with self._autocast():
                        model_output = self.model(batch_features)
                        loss, loss_statistics = combined_loss(
                            model_output,
                            batch_targets,
                            loss_stage,
                            self.direct_loss_weights,
                            return_statistics=True,
                        )

                    self.scaler.scale(
                        loss
                    ).backward()  # pyright: ignore[reportUnknownMemberType]
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), GRAD_CLIP_NORM
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    loss_parts, grad_norm_value = (
                        loss_statistics.materialize(grad_norm)
                    )
                    loss_parts["step"] = self.state.finish_step()

                    metrics.update(
                        rows=batch_rows,
                        loss_parts=loss_parts,
                        grad_norm=grad_norm_value,
                        **missingness_ratios,
                    )
                    metrics.train_step_ms += (
                        time.perf_counter() - phase_started
                    ) * 1000
            finally:
                if isinstance(batches, _Closable):
                    batches.close()

        metrics.elapsed_ms = (time.perf_counter() - started) * 1000
        return metrics

    def _train_loader(self, loader: Iterable[TensorBatch]) -> TrainMetrics:
        return self._train_loaders((loader,))

    def _observe_metrics(
        self,
        metrics: TrainMetrics,
        frame: int | None = None,
        epoch: int | None = None,
    ) -> SelectionPayload:
        checkpoint_best = False
        should_stop = False
        selection_score = None
        maximum_stage_epoch = (
            metrics.minimum_loss_stage == self.loss_stage
            and metrics.maximum_loss_stage == self.loss_stage
        )
        if maximum_stage_epoch:
            self.maximum_stage_completed = True
            if self.selection_state is not None:
                if not self.selection_state.active:
                    self._reset_selection()
                components = metrics.direct_losses()
                selection_score = math.fsum(
                    weight * value
                    for weight, value in zip(
                        self.direct_loss_weights,
                        components,
                        strict=True,
                    )
                )
                if not math.isfinite(selection_score):
                    raise ValueError("checkpoint selection score must be finite")
                checkpoint_best, should_stop = self.selection_state.update(
                    selection_score
                )
                metrics.selection_score = selection_score
                if checkpoint_best:
                    self.best_selection_score = selection_score
                    self.best_state_dict = self._snapshot_state_dict()
                    self.best_metrics = metrics.to_dict()
                    self.best_frame = frame
                    self.best_epoch = epoch

        return {
            "selection_score": selection_score,
            "checkpoint_best": checkpoint_best,
            "should_stop": should_stop,
            "best_selection_score": (
                self.best_selection_score
                if math.isfinite(self.best_selection_score)
                else None
            ),
        }

    def _reset_selection(self) -> None:
        if self.selection_state is None:
            return
        self.selection_state.begin()
        self.best_selection_score = float("inf")
        self.best_state_dict = None
        self.best_metrics = None
        self.best_frame = None
        self.best_epoch = None

    def _snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _restore_best_state_dict(self) -> None:
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
        features: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
    ) -> TrainMetrics:
        loader = self._data_loader(features, targets)

        self.state.begin_epoch(epoch)
        return self._train_loader(loader)

    def fit_epochs(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        on_epoch: EpochCallback | None = None,
        frame: int | None = None,
    ) -> list[TrainMetrics]:
        loader = self._data_loader(features, targets)
        return self._fit_loader_epochs(
            lambda: (loader,),
            on_epoch=on_epoch,
            frame=frame,
        )

    def fit_payloads(
        self,
        payloads: PayloadFactory,
        on_epoch: EpochCallback | None = None,
    ) -> list[TrainMetrics]:
        """Train global epochs over a payload-independent row stream.

        ``payloads`` is a callable so durable inputs can be reopened for every
        epoch. Optimizer batches and bounded shuffle windows may cross payload
        boundaries, so transport partitioning cannot change the trajectory.
        """
        def loaders() -> Iterator[Iterable[TensorBatch]]:
            yield self._prefetched_payload_batches(payloads())

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
        )

    def _payload_batches(
        self,
        payloads: Payloads,
        generator: torch.Generator,
    ) -> Iterator[TensorBatch]:
        window_rows: int | None = None
        source_buffer: torch.Tensor | None = None
        target_buffer: torch.Tensor | None = None
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

            if window_rows is None or target_buffer is None:
                raise AssertionError("shuffle buffers were not initialized")

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
            if source_buffer is None or target_buffer is None:
                raise AssertionError("shuffle buffers were not initialized")
            yield from self._shuffled_batches(
                source_buffer,
                target_buffer,
                buffered_rows,
                generator,
            )

    def _prefetched_payload_batches(
        self,
        payloads: Payloads,
    ) -> _BatchPrefetcher:
        return _BatchPrefetcher(self._payload_batches(
            payloads,
            self._payload_shuffle_generator,
        ))

    def _shuffled_batches(
        self,
        source: torch.Tensor,
        targets: torch.Tensor,
        rows: int,
        generator: torch.Generator,
    ) -> Iterator[TensorBatch]:
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

    def _data_loader(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> Iterable[TensorBatch]:
        dataset = TensorDataset(features, targets)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def _fit_loader_epochs(
        self,
        loaders: Callable[[], Iterable[Iterable[TensorBatch]]],
        on_epoch: EpochCallback | None = None,
        frame: int | None = None,
        *,
        start_epoch: int = 0,
        on_epoch_committed: EpochCommittedCallback | None = None,
    ) -> list[TrainMetrics]:
        metrics_rows: list[TrainMetrics] = []
        self.state.begin_frame(frame)
        self.training_complete = False

        for epoch in range(start_epoch, self.epochs):
            self.state.begin_epoch(epoch)
            metrics = self._train_loaders(loaders())
            metrics_rows.append(metrics)
            selection_payload = self._observe_metrics(
                metrics,
                frame=frame,
                epoch=epoch + 1,
            )

            if on_epoch is not None:
                on_epoch(epoch, metrics, selection_payload)

            should_stop = selection_payload["should_stop"]
            self.state.finish_epoch()
            self.training_complete = (
                should_stop or self.state.global_epoch >= self.epochs
            )
            if self.training_complete and not self.maximum_stage_completed:
                raise ValueError(
                    "training completed before the maximum loss stage"
                )
            if on_epoch_committed is not None:
                on_epoch_committed(
                    epoch,
                    metrics,
                    selection_payload,
                    self.training_complete,
                )
            if should_stop:
                break

        return metrics_rows

    def fit_payloads_resumable(
        self,
        payloads: PayloadFactory,
        *,
        on_epoch: EpochCallback | None = None,
        on_epoch_committed: EpochCommittedCallback | None = None,
    ) -> list[TrainMetrics]:
        """Train job-wide epochs and expose only complete recovery boundaries."""

        def loaders() -> Iterator[Iterable[TensorBatch]]:
            yield self._prefetched_payload_batches(payloads())

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
            on_epoch_committed=on_epoch_committed,
        )

    def fit_streaming_payloads(
        self,
        first_epoch_payloads: Payloads,
        closed_payloads: PayloadFactory,
        *,
        on_epoch: EpochCallback | None = None,
        on_epoch_committed: EpochCommittedCallback | None = None,
    ) -> list[TrainMetrics]:
        """Train epoch zero from an open stream, then replay closed input.

        The first iterable may block at the durable input frontier. Its EOF is
        the explicit input-close boundary. Every later epoch reopens the same
        complete ordered dataset through ``closed_payloads``.
        """

        first_epoch = True

        def loaders() -> Iterator[Iterable[TensorBatch]]:
            nonlocal first_epoch
            if first_epoch:
                payloads = first_epoch_payloads
                first_epoch = False
                batches = self._payload_batches(
                    payloads,
                    self._payload_shuffle_generator,
                )
            else:
                payloads = closed_payloads()
                batches = self._prefetched_payload_batches(payloads)
            yield batches

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
            on_epoch_committed=on_epoch_committed,
        )

    def recovery_state_dict(self) -> dict[str, object]:
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
            "objective_config_sha256": self._objective_config_sha256(),
            "selection_state": (
                None
                if self.selection_state is None
                else asdict(self.selection_state)
            ),
            "selection": {
                "best_selection_score": self.best_selection_score,
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
            "maximum_stage_completed": self.maximum_stage_completed,
        }

    def load_recovery_state_dict(
        self,
        payload: Mapping[str, object],
    ) -> None:
        """Restore a state produced by :meth:`recovery_state_dict`."""

        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
            "training_state",
            "objective_config_sha256",
            "selection_state",
            "selection",
            "payload_shuffle_generator_state",
            "rng",
            "training_complete",
            "maximum_stage_completed",
        }
        if set(payload) != required:
            raise ValueError("training recovery state has invalid fields")

        training_state = _object_dict(
            payload["training_state"],
            "training recovery progress",
        )
        selection_state_value = payload["selection_state"]
        selection = _object_dict(
            payload["selection"],
            "training recovery checkpoint selection",
        )
        rng = _object_dict(payload["rng"], "training recovery random state")
        if set(training_state) != {
            "frame",
            "frame_epoch",
            "global_epoch",
            "train_step",
        }:
            raise ValueError("training recovery progress is invalid")
        if payload["objective_config_sha256"] != self._objective_config_sha256():
            raise ValueError("training recovery objective configuration differs")
        if self.selection is None:
            if selection_state_value is not None:
                raise ValueError("training recovery selection state is invalid")
            selection_state = None
        else:
            selection_state = _object_dict(
                selection_state_value,
                "training recovery selection state",
            )
            if set(selection_state) != {
                "min_delta",
                "patience",
                "active",
                "best_score",
                "wait",
            }:
                raise ValueError("training recovery selection state is invalid")
        if set(selection) != {
            "best_selection_score",
            "best_state_dict",
            "best_metrics",
            "best_frame",
            "best_epoch",
        }:
            raise ValueError("training recovery checkpoint selection is invalid")
        if set(rng) != {
            "python",
            "numpy",
            "torch",
            "cuda",
        }:
            raise ValueError("training recovery random state is invalid")
        if self.selection is not None:
            if selection_state is None:
                raise AssertionError("selection state was not initialized")
            if (
                _number(selection_state["min_delta"], "selection min_delta")
                != self.selection.min_delta
                or _integer(selection_state["patience"], "selection patience")
                != self.selection.patience
            ):
                raise ValueError("training recovery selection policy differs")

        self.model.load_state_dict(
            _tensor_state_dict(
                payload["model_state_dict"],
                "training recovery model state",
            )
        )
        optimizer_state = _object_dict(
            payload["optimizer_state_dict"],
            "training recovery optimizer state",
        )
        self.optimizer.load_state_dict(optimizer_state)
        _optimizer_to(self.optimizer, self.device)
        scaler_state = _object_dict(
            payload["scaler_state_dict"],
            "training recovery scaler state",
        )
        self.scaler.load_state_dict(scaler_state)
        self.state = TrainingState(
            frame=_integer(training_state["frame"], "training frame"),
            frame_epoch=_integer(
                training_state["frame_epoch"],
                "training frame epoch",
            ),
            global_epoch=_integer(
                training_state["global_epoch"],
                "training global epoch",
            ),
            train_step=_integer(
                training_state["train_step"],
                "training step",
            ),
        )
        self.selection_state = (
            None
            if selection_state is None
            else SelectionState(
                min_delta=_number(
                    selection_state["min_delta"],
                    "selection min_delta",
                ),
                patience=_integer(
                    selection_state["patience"],
                    "selection patience",
                ),
                active=_boolean(
                    selection_state["active"],
                    "selection active",
                ),
                best_score=_number(
                    selection_state["best_score"],
                    "selection best score",
                ),
                wait=_integer(selection_state["wait"], "selection wait"),
            )
        )
        self.best_selection_score = _number(
            selection["best_selection_score"],
            "best selection score",
        )
        self.best_state_dict = _optional_tensor_state_dict(
            selection["best_state_dict"],
            "best model state",
        )
        self.best_metrics = _optional_json_object(selection["best_metrics"])
        self.best_frame = _optional_integer(
            selection["best_frame"],
            "best frame",
        )
        self.best_epoch = _optional_integer(
            selection["best_epoch"],
            "best epoch",
        )
        shuffle_state = payload["payload_shuffle_generator_state"]
        if not isinstance(shuffle_state, torch.Tensor):
            raise ValueError("training recovery shuffle state is invalid")
        self._payload_shuffle_generator.set_state(
            shuffle_state
        )
        random.setstate(cast(tuple[object, ...], rng["python"]))
        np.random.set_state(
            cast(
                tuple[str, np.ndarray, int, int, float],
                rng["numpy"],
            )
        )
        torch_rng_state = rng["torch"]
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("training recovery Torch RNG state is invalid")
        torch.set_rng_state(torch_rng_state)
        cuda_rng_state = rng["cuda"]
        if cuda_rng_state is not None:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError(
                    "CUDA training recovery requires an available CUDA device"
                )
            if not isinstance(cuda_rng_state, list):
                raise ValueError("training recovery CUDA RNG state is invalid")
            cuda_states = cast(list[object], cuda_rng_state)
            if not all(isinstance(state, torch.Tensor) for state in cuda_states):
                raise ValueError("training recovery CUDA RNG state is invalid")
            torch.cuda.set_rng_state_all(
                cast(list[torch.Tensor], cuda_states)
            )
        self.training_complete = _boolean(
            payload["training_complete"],
            "training completion marker",
        )
        self.maximum_stage_completed = _boolean(
            payload["maximum_stage_completed"],
            "maximum stage completion marker",
        )

    def fit(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        model_name: str,
    ) -> None:
        def on_epoch(
            epoch: int,
            metrics: TrainMetrics,
            selection_payload: SelectionPayload,
        ) -> None:
            stats = tree_stats(self.model.parameters())

            print(metrics.console_line(
                epoch=epoch + 1,
                norm=f"{stats['norm']:.0f}",
                **self.metrics_context,
                **selection_payload,
            ))
            self.record_metrics(
                metrics,
                mode="fit",
                epoch=epoch + 1,
                norm=stats["norm"],
                selection_score=selection_payload["selection_score"],
                checkpoint_best=selection_payload["checkpoint_best"],
                should_stop=selection_payload["should_stop"],
                best_selection_score=selection_payload[
                    "best_selection_score"
                ],
            )

        print(self.config_line())
        self.fit_epochs(features, targets, on_epoch=on_epoch)

        self.save(model_name)
        print("Model saved")

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            if features.size(0) == 0:
                return public_predictions(self.model(features.to(self.device)))

            predictions = None
            for offset in range(0, features.size(0), self.batch_size):
                batch = features[offset:offset + self.batch_size].to(self.device)
                output = public_predictions(self.model(batch))
                if predictions is None:
                    predictions = torch.empty(
                        (features.size(0), *output.shape[1:]),
                        dtype=output.dtype,
                        device=output.device,
                    )
                predictions[offset:offset + output.size(0)].copy_(output)

            if predictions is None:
                raise AssertionError("prediction buffer was not initialized")
            return predictions

    def load(self, model_name: str) -> None:
        load_model(model_name, self.model, self.device)

    def load_payload(self, checkpoint: dict[str, object]) -> None:
        self.model.load_state_dict(
            _tensor_state_dict(checkpoint.get("state_dict"), "checkpoint state")
        )

    def save(self, model_name: str) -> None:
        if self.selection is not None and self.best_state_dict is not None:
            self._restore_best_state_dict()

        save_model(
            model_name,
            self.model,
            model_config=self.model_config,
            train_config=self.train_config,
            data_contract=self.data_contract,
            extra={
                "checkpoint_selection": {
                    "enabled": self.selection is not None,
                    "objective_config_sha256": self._objective_config_sha256(),
                    "best_selection_score": (
                        self.best_selection_score
                        if math.isfinite(self.best_selection_score)
                        else None
                    ),
                    "best_frame": self.best_frame,
                    "best_epoch": self.best_epoch,
                    "source": (
                        "best_selection_score"
                        if self.best_state_dict is not None
                        else "last_maximum_stage"
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
            "selection": "on" if self.selection is not None else "off",
            "context_mode": self.context_mode,
            "amp": self.use_amp,
        })
        return "config " + " ".join(
            f"{key}={value}" for key, value in fields.items()
        )

    def _objective_config_sha256(self) -> str:
        if self.train_config is None:
            raise ValueError("training configuration is unavailable")
        return objective_config_sha256(self.train_config)

    def record_metrics(
        self,
        metrics: TrainMetrics,
        **extra: JsonValue,
    ) -> None:
        payload: dict[str, JsonValue] = {
            **self.metrics_context,
            **extra,
        }
        payload.setdefault("context_mode", self.context_mode)
        append_metrics_jsonl(self.metrics_path, metrics, **payload)


def _object_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


def _tensor_state_dict(
    value: object,
    label: str,
) -> dict[str, torch.Tensor]:
    mapping = _object_dict(value, label)
    if not all(isinstance(item, torch.Tensor) for item in mapping.values()):
        raise ValueError(f"{label} must contain only tensors")
    return cast(dict[str, torch.Tensor], mapping)


def _optional_tensor_state_dict(
    value: object,
    label: str,
) -> dict[str, torch.Tensor] | None:
    if value is None:
        return None
    return _tensor_state_dict(value, label)


def _optional_json_object(value: object) -> JsonObject | None:
    if value is None:
        return None
    return cast(JsonObject, _object_dict(value, "best metrics"))


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _tree_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            key: _tree_to_cpu(item)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return tuple(
            _tree_to_cpu(item)
            for item in cast(tuple[object, ...], value)
        )
    return copy.deepcopy(value)


def _optimizer_to(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    states = cast(
        Mapping[object, dict[object, object]],
        optimizer.state,
    )
    for state in states.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
