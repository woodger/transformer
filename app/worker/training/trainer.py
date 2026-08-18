from __future__ import annotations

import copy
import math
import random
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from typing import TypedDict, cast

import numpy as np
import torch

from app.contracts.json_types import JsonValue
from app.contracts.worker.v6.config import (
    DEFAULT_CONTEXT_MODE,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v6.objective import objective_config_sha256
from app.worker.checkpoints.model import load_model, save_model
from app.worker.data.tensors import TrainingBatch
from app.worker.model.context import context_missingness_ratios
from app.worker.model.transformer import public_predictions
from app.worker.telemetry import (
    EpochTelemetry,
    ObservedTrainingEpoch,
    TargetErrorObservation,
    append_epoch_telemetry,
    format_epoch_console_line,
)
from app.worker.training.batching import Closable, PayloadBatcher, TrainingBatches
from app.worker.training.constants import GRAD_CLIP_NORM
from app.worker.training.early_stopping import SelectionState
from app.worker.training.loss_scheduler import LossScheduler
from app.worker.training.losses import (
    combined_loss,
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)
from app.worker.training.parameter_stats import parameter_tree_stats
from app.worker.training.training_state import TrainingState

TrainingBatchFactory = Callable[[], TrainingBatches]


class SelectionPayload(TypedDict):
    selection_score: float | None
    checkpoint_best: bool
    should_stop: bool
    best_selection_score: float | None


EpochCallback = Callable[[int, ObservedTrainingEpoch, SelectionPayload], None]
EpochCommittedCallback = Callable[
    [int, ObservedTrainingEpoch, SelectionPayload, bool],
    None,
]


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
        train_config: TrainConfig,
        *,
        metrics_path: str | None = None,
        context_mode: str = DEFAULT_CONTEXT_MODE,
        metrics_context: Mapping[str, JsonValue] | None = None,
        model_config: ModelConfig | None = None,
        data_contract: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.train_config = train_config
        self.batch_size = train_config.batch_size
        self._batcher = PayloadBatcher(self.batch_size)
        self.epochs = train_config.epochs
        self.loss_stage = validate_loss_stage(train_config.loss_stage)
        self.loss_schedule = validate_loss_schedule(
            train_config.loss_schedule
        )
        self.stage_size = validate_stage_size(train_config.stage_size)
        self.direct_loss_weights = train_config.direct_loss_weights
        self.selection = train_config.selection
        self.metrics_path = metrics_path
        self.context_mode = context_mode
        self.model_config = model_config
        self.data_contract = (
            None if data_contract is None else dict(data_contract)
        )
        self.seed = train_config.seed
        self.best_selection_score: float = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_frame: int | None = None
        self.best_epoch: int | None = None
        self.state = TrainingState()
        self.selection_state = (
            None
            if self.selection is None
            else SelectionState(
                self.selection.min_delta,
                self.selection.patience,
            )
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

        self.use_amp = bool(train_config.use_amp and device.type == "cuda")
        self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self._optimizer_updates_applied_total = 0
        self.optimizer.register_step_post_hook(
            self._record_optimizer_update,
        )

        if train_config.use_amp and device.type != "cuda":
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

    def _record_optimizer_update(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        self._optimizer_updates_applied_total += 1

    def _train_loaders(
        self,
        loaders: Iterable[TrainingBatches],
    ) -> ObservedTrainingEpoch:
        self.model.train()
        epoch_result = ObservedTrainingEpoch(
            lr=self.optimizer.param_groups[0]["lr"],
            step=self.state.train_step,
            loss_stage=self._loss_stage_for(),
            telemetry=EpochTelemetry(),
        )
        started = time.perf_counter()

        # Phase timers stay host-side and must never force CUDA synchronization.
        for loader in loaders:
            batches = iter(loader)
            try:
                while True:
                    phase_started = time.perf_counter()
                    try:
                        batch = next(batches)
                    except StopIteration:
                        break
                    telemetry = epoch_result.telemetry
                    if telemetry is not None:
                        telemetry.input_pipeline_ms += (
                            time.perf_counter() - phase_started
                        ) * 1000

                    loss_stage = self._loss_stage_for()
                    batch_rows = batch.features.size(0)
                    phase_started = time.perf_counter()
                    missingness_ratios: dict[str, float] = {}
                    if telemetry is not None:
                        try:
                            missingness_ratios = context_missingness_ratios(
                                batch.features,
                                self.context_mode,
                            )
                            telemetry.missing_stats_ms += (
                                time.perf_counter() - phase_started
                            ) * 1000
                        except Exception as exc:
                            self._disable_epoch_telemetry(epoch_result, exc)

                    phase_started = time.perf_counter()
                    batch_features = batch.features.to(self.device)
                    batch_targets = batch.targets.to(self.device)
                    telemetry = epoch_result.telemetry
                    if telemetry is not None:
                        telemetry.host_to_device_ms += (
                            time.perf_counter() - phase_started
                        ) * 1000

                    phase_started = time.perf_counter()
                    self.optimizer.zero_grad()

                    with self._autocast():
                        model_output = self.model(batch_features)
                        loss_evaluation = combined_loss(
                            model_output,
                            batch_targets,
                            loss_stage,
                            self.direct_loss_weights,
                            return_statistics=True,
                        )
                        loss = loss_evaluation.loss

                    target_error_observation: TargetErrorObservation | None = None
                    if epoch_result.telemetry is not None:
                        try:
                            target_error_observation = (
                                TargetErrorObservation.evaluate(
                                    model_output,
                                    batch_targets,
                                )
                            )
                        except Exception as exc:
                            self._disable_epoch_telemetry(epoch_result, exc)

                    self.scaler.scale(
                        loss
                    ).backward()  # pyright: ignore[reportUnknownMemberType]
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), GRAD_CLIP_NORM
                    )
                    updates_before = self._optimizer_updates_applied_total
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    updates_after = self._optimizer_updates_applied_total
                    optimizer_update_applied = updates_after > updates_before
                    try:
                        materialized_statistics = (
                            loss_evaluation.statistics.materialize(
                                (
                                    grad_norm
                                    if target_error_observation is not None
                                    else None
                                ),
                                (
                                    ()
                                    if target_error_observation is None
                                    else target_error_observation.values
                                ),
                            )
                        )
                    except Exception as exc:
                        if target_error_observation is None:
                            raise
                        self._disable_epoch_telemetry(epoch_result, exc)
                        target_error_observation = None
                        materialized_statistics = (
                            loss_evaluation.statistics.materialize()
                        )
                    loss_parts = materialized_statistics.parts
                    grad_norm_value = materialized_statistics.grad_norm
                    loss_parts["step"] = self.state.finish_step()

                    epoch_result.update(batch_rows, loss_parts)
                    telemetry = epoch_result.telemetry
                    if telemetry is not None and target_error_observation is not None:
                        try:
                            target_errors = target_error_observation.decode(
                                materialized_statistics.observations
                            )
                            telemetry.observe_batch(
                                rows=batch_rows,
                                target_errors=target_errors,
                                grad_norm=grad_norm_value,
                                optimizer_update_applied=(
                                    optimizer_update_applied
                                ),
                                amp_overflow=(
                                    self.use_amp and not optimizer_update_applied
                                ),
                                **missingness_ratios,
                            )
                            telemetry.train_step_ms += (
                                time.perf_counter() - phase_started
                            ) * 1000
                        except Exception as exc:
                            self._disable_epoch_telemetry(epoch_result, exc)
            finally:
                if isinstance(batches, Closable):
                    batches.close()

        telemetry = epoch_result.telemetry
        if telemetry is not None:
            telemetry.elapsed_ms = (time.perf_counter() - started) * 1000
            telemetry.finalize_gradient_statistics()
        return epoch_result

    @staticmethod
    def _disable_epoch_telemetry(
        epoch_result: ObservedTrainingEpoch,
        exc: Exception,
    ) -> None:
        if epoch_result.telemetry is None:
            return
        epoch_result.telemetry = None
        print(
            "training epoch telemetry disabled: " + type(exc).__name__,
            file=sys.stderr,
            flush=True,
        )

    def _train_loader(self, loader: TrainingBatches) -> ObservedTrainingEpoch:
        return self._train_loaders((loader,))

    def _observe_metrics(
        self,
        metrics: ObservedTrainingEpoch,
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
                selection_decision = self.selection_state.update(
                    selection_score,
                )
                checkpoint_best = selection_decision.improved
                should_stop = selection_decision.should_stop
                metrics.selection_score = selection_score
                if checkpoint_best:
                    self.best_selection_score = selection_score
                    self.best_state_dict = self._snapshot_state_dict()
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
        batch: TrainingBatch,
        epoch: int = 0,
    ) -> ObservedTrainingEpoch:
        loader = self._batcher.data_loader(batch)

        self.state.begin_epoch(epoch)
        return self._train_loader(loader)

    def fit_epochs(
        self,
        batch: TrainingBatch,
        on_epoch: EpochCallback | None = None,
        frame: int | None = None,
    ) -> list[ObservedTrainingEpoch]:
        loader = self._batcher.data_loader(batch)
        return self._fit_loader_epochs(
            lambda: (loader,),
            on_epoch=on_epoch,
            frame=frame,
        )

    def fit_payloads(
        self,
        payloads: TrainingBatchFactory,
        on_epoch: EpochCallback | None = None,
    ) -> list[ObservedTrainingEpoch]:
        """Train global epochs over a payload-independent row stream.

        ``payloads`` is a callable so durable inputs can be reopened for every
        epoch. Optimizer batches and bounded shuffle windows may cross payload
        boundaries, so transport partitioning cannot change the trajectory.
        """
        def loaders() -> Iterator[TrainingBatches]:
            yield self._batcher.prefetched(
                payloads(),
                self._payload_shuffle_generator,
            )

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
        )

    def _fit_loader_epochs(
        self,
        loaders: Callable[[], Iterable[TrainingBatches]],
        on_epoch: EpochCallback | None = None,
        frame: int | None = None,
        *,
        start_epoch: int = 0,
        on_epoch_committed: EpochCommittedCallback | None = None,
    ) -> list[ObservedTrainingEpoch]:
        metrics_rows: list[ObservedTrainingEpoch] = []
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
        payloads: TrainingBatchFactory,
        *,
        on_epoch: EpochCallback | None = None,
        on_epoch_committed: EpochCommittedCallback | None = None,
    ) -> list[ObservedTrainingEpoch]:
        """Train job-wide epochs and expose only complete recovery boundaries."""

        def loaders() -> Iterator[TrainingBatches]:
            yield self._batcher.prefetched(
                payloads(),
                self._payload_shuffle_generator,
            )

        return self._fit_loader_epochs(
            loaders,
            on_epoch=on_epoch,
            start_epoch=self.state.global_epoch,
            on_epoch_committed=on_epoch_committed,
        )

    def fit_streaming_payloads(
        self,
        first_epoch_payloads: TrainingBatches,
        closed_payloads: TrainingBatchFactory,
        *,
        on_epoch: EpochCallback | None = None,
        on_epoch_committed: EpochCommittedCallback | None = None,
    ) -> list[ObservedTrainingEpoch]:
        """Train epoch zero from an open stream, then replay closed input.

        The first iterable may block at the durable input frontier. Its EOF is
        the explicit input-close boundary. Every later epoch reopens the same
        complete ordered dataset through ``closed_payloads``.
        """

        first_epoch = True

        def loaders() -> Iterator[TrainingBatches]:
            nonlocal first_epoch
            if first_epoch:
                payloads = first_epoch_payloads
                first_epoch = False
                batches = self._batcher.batches(
                    payloads,
                    self._payload_shuffle_generator,
                )
            else:
                payloads = closed_payloads()
                batches = self._batcher.prefetched(
                    payloads,
                    self._payload_shuffle_generator,
                )
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
                # Kept as an inert field to preserve the current durable
                # recovery format. Telemetry is no longer restored into the
                # training state.
                "best_metrics": None,
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
        _validate_legacy_best_metrics(selection["best_metrics"])
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
        batch: TrainingBatch,
        model_name: str,
    ) -> None:
        def on_epoch(
            epoch: int,
            metrics: ObservedTrainingEpoch,
            selection_payload: SelectionPayload,
        ) -> None:
            stats = parameter_tree_stats(self.model.parameters())

            print(format_epoch_console_line(
                metrics,
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
        self.fit_epochs(batch, on_epoch=on_epoch)

        self.save(model_name)
        print("Model saved")

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            if features.size(0) == 0:
                return public_predictions(self.model(features.to(self.device)))

            predictions = None
            for offset in range(0, features.size(0), self.batch_size):
                batch_features = features[
                    offset:offset + self.batch_size
                ].to(self.device)
                batch_predictions = public_predictions(self.model(batch_features))
                if predictions is None:
                    predictions = torch.empty(
                        (features.size(0), *batch_predictions.shape[1:]),
                        dtype=batch_predictions.dtype,
                        device=batch_predictions.device,
                    )
                predictions[
                    offset:offset + batch_predictions.size(0)
                ].copy_(batch_predictions)

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
        return objective_config_sha256(self.train_config)

    def record_metrics(
        self,
        metrics: ObservedTrainingEpoch,
        **extra: JsonValue,
    ) -> None:
        payload: dict[str, JsonValue] = {
            **self.metrics_context,
            **extra,
        }
        payload.setdefault("context_mode", self.context_mode)
        append_epoch_telemetry(self.metrics_path, metrics, **payload)


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


def _validate_legacy_best_metrics(value: object) -> None:
    if value is not None and not isinstance(value, dict):
        raise ValueError("best metrics must be an object or null")




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
