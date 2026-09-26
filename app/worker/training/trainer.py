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

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v5 import ModelContract
from app.contracts.worker.v20.config import (
    DEFAULT_CONTEXT_MODE,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v20.diagnostics import (
    ENCODER_LAYER_DIRECT_COMPONENT_PER_BATCH,
    TARGET_HEAD_FULL_COMMITTED_ARTIFACT,
)
from app.worker.checkpoints.model import save_model
from app.worker.data.tensors import TrainingBatch
from app.worker.model.context import context_missingness_ratios
from app.worker.model.transformer import public_predictions
from app.worker.telemetry import (
    EpochTelemetry,
    GradientInteractionObservation,
    ObservedTrainingEpoch,
    TargetErrorObservation,
)
from app.worker.telemetry.target_head import TargetHeadDiagnosticsCollector
from app.worker.training.batching import Closable, PayloadBatcher, TrainingBatches
from app.worker.training.constants import GRAD_CLIP_NORM
from app.worker.training.early_stopping import SelectionState
from app.worker.training.losses import combined_loss
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
        model_contract: ModelContract,
        *,
        context_mode: str = DEFAULT_CONTEXT_MODE,
        model_config: ModelConfig | None = None,
        model_definition_sha256: str,
        initialization: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.train_config = train_config
        self.model_contract = model_contract
        self.batch_size = train_config.batch_size
        self._batcher = PayloadBatcher(self.batch_size)
        self.epochs = train_config.epochs
        self.direct_loss_weights = model_contract.direct_loss_weights
        self.selection = train_config.selection
        self.context_mode = context_mode
        self.model_config = model_config
        self.model_definition_sha256 = model_definition_sha256
        self.initialization: JsonObject | None = (
            None
            if initialization is None
            else cast(JsonObject, dict(initialization))
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
        self.training_complete = False
        self._payload_shuffle_generator = torch.Generator()
        self._payload_shuffle_generator.manual_seed(self.seed)
        self._target_head_diagnostics = (
            None
            if train_config.diagnostics.target_head
            != TARGET_HEAD_FULL_COMMITTED_ARTIFACT
            else TargetHeadDiagnosticsCollector(
                model_contract,
                collect_encoder_learning=(
                    train_config.diagnostics.encoder_layer_diagnostics
                    == ENCODER_LAYER_DIRECT_COMPONENT_PER_BATCH
                ),
            )
        )

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

    @property
    def target_head_diagnostics_enabled(self) -> bool:
        return self._target_head_diagnostics is not None

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
        self._begin_target_head_diagnostics_epoch()
        self.model.train()
        epoch_result = ObservedTrainingEpoch(
            targets=self.model_contract.target_identities,
            direct_components=tuple(
                (str(item["identity"]), str(item["operator"]))
                for item in self.model_contract.direct_components
            ),
            auxiliary_components=tuple(
                (str(item["identity"]), str(item["operator"]))
                for item in self.model_contract.auxiliary_components
            ),
            lr=self.optimizer.param_groups[0]["lr"],
            step=self.state.train_step,
            telemetry=EpochTelemetry(
                targets=self.model_contract.target_identities,
                component_targets={
                    str(item["identity"]): (
                        self.model_contract.target_identities[index],
                        index,
                    )
                    for index, item in enumerate(
                        self.model_contract.direct_components
                    )
                },
            ),
        )
        started = time.perf_counter()

        # Таймеры фаз работают на CPU и не должны принудительно синхронизировать CUDA.
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

                    batch_rows = batch.features.size(0)
                    phase_started = time.perf_counter()
                    missingness_ratios = self._collect_missingness_ratios(
                        epoch_result,
                        batch.features,
                        phase_started,
                    )

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

                    gradient_observation: GradientInteractionObservation | None = None
                    shared_representation: torch.Tensor | None = None
                    sample_gradient_interactions = (
                        epoch_result.telemetry is not None
                        and self._sample_gradient_interactions(
                            epoch_result.telemetry.gradient_interaction_samples
                        )
                    )
                    raw_forward = getattr(
                        self.model,
                        "forward_with_shared_representation",
                        None,
                    )
                    if sample_gradient_interactions and not callable(raw_forward):
                        self._disable_epoch_telemetry(
                            epoch_result,
                            ValueError(
                                "gradient diagnostics require a Transformer model"
                            ),
                        )
                        sample_gradient_interactions = False

                    with self._autocast():
                        if sample_gradient_interactions:
                            forward = cast(
                                Callable[
                                    [torch.Tensor],
                                    tuple[torch.Tensor, torch.Tensor],
                                ],
                                raw_forward,
                            )
                            model_output, shared_representation = forward(
                                batch_features
                            )
                        else:
                            model_output = self.model(batch_features)
                            shared_representation = None
                        loss_evaluation = combined_loss(
                            model_output,
                            batch_targets,
                            self.model_contract,
                            return_statistics=True,
                        )
                        loss = loss_evaluation.loss

                    if shared_representation is not None:
                        try:
                            gradient_observation = GradientInteractionObservation.evaluate(
                                loss_evaluation.diagnostic_components,
                                shared_representation,
                            )
                        except Exception as exc:
                            self._disable_epoch_telemetry(epoch_result, exc)

                    self._record_target_head_component_gradients(
                        loss_evaluation.diagnostic_components,
                    )

                    target_error_observation = self._observe_target_errors(
                        epoch_result,
                        model_output,
                        batch_targets,
                    )

                    self.scaler.scale(
                        loss
                    ).backward()  # pyright: ignore[reportUnknownMemberType]
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), GRAD_CLIP_NORM
                    )
                    updates_before = self._optimizer_updates_applied_total
                    self._snapshot_encoder_layer_parameters()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    updates_after = self._optimizer_updates_applied_total
                    optimizer_update_applied = updates_after > updates_before
                    self._record_encoder_layer_parameter_updates(
                        optimizer_update_applied,
                    )
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
                    grad_norm_value = materialized_statistics.grad_norm
                    step = self.state.finish_step()

                    epoch_result.update(
                        batch_rows,
                        materialized_statistics,
                        step=step,
                    )
                    self._record_batch_telemetry(
                        epoch_result,
                        batch_rows=batch_rows,
                        target_error_observation=target_error_observation,
                        observations=materialized_statistics.observations,
                        grad_norm=grad_norm_value,
                        optimizer_update_applied=optimizer_update_applied,
                        missingness_ratios=missingness_ratios,
                        gradient_observation=gradient_observation,
                        phase_started=phase_started,
                    )
            finally:
                if isinstance(batches, Closable):
                    batches.close()

        telemetry = epoch_result.telemetry
        if telemetry is not None:
            telemetry.elapsed_ms = (time.perf_counter() - started) * 1000
            telemetry.finalize_gradient_statistics()
        return epoch_result

    def _collect_missingness_ratios(
        self,
        epoch_result: ObservedTrainingEpoch,
        features: torch.Tensor,
        phase_started: float,
    ) -> dict[str, float]:
        telemetry = epoch_result.telemetry
        if telemetry is None:
            return {}
        try:
            ratios = context_missingness_ratios(
                features,
                self.context_mode,
            )
            telemetry.missing_stats_ms += (
                time.perf_counter() - phase_started
            ) * 1000
            return ratios
        except Exception as exc:
            self._disable_epoch_telemetry(epoch_result, exc)
            return {}

    def _observe_target_errors(
        self,
        epoch_result: ObservedTrainingEpoch,
        model_output: torch.Tensor,
        batch_targets: torch.Tensor,
    ) -> TargetErrorObservation | None:
        if epoch_result.telemetry is None:
            return None
        try:
            return TargetErrorObservation.evaluate(
                model_output,
                batch_targets,
                self.model_contract,
            )
        except Exception as exc:
            self._disable_epoch_telemetry(epoch_result, exc)
            return None

    def _record_batch_telemetry(
        self,
        epoch_result: ObservedTrainingEpoch,
        *,
        batch_rows: int,
        target_error_observation: TargetErrorObservation | None,
        observations: tuple[float, ...],
        grad_norm: float | None,
        optimizer_update_applied: bool,
        missingness_ratios: dict[str, float],
        gradient_observation: GradientInteractionObservation | None,
        phase_started: float,
    ) -> None:
        telemetry = epoch_result.telemetry
        if telemetry is None or target_error_observation is None:
            return
        try:
            target_errors = target_error_observation.decode(observations)
            telemetry.observe_batch(
                rows=batch_rows,
                target_errors=target_errors,
                grad_norm=grad_norm,
                optimizer_update_applied=optimizer_update_applied,
                amp_overflow=self.use_amp and not optimizer_update_applied,
                **missingness_ratios,
            )
            if gradient_observation is not None:
                telemetry.observe_gradient_interactions(
                    gradient_observation.materialize()
                )
            telemetry.train_step_ms += (
                time.perf_counter() - phase_started
            ) * 1000
        except Exception as exc:
            self._disable_epoch_telemetry(epoch_result, exc)

    def _begin_target_head_diagnostics_epoch(self) -> None:
        collector = self._target_head_diagnostics
        if collector is None:
            return
        try:
            collector.begin_epoch(self.model)
        except Exception as exc:
            self._disable_target_head_diagnostics(collector, exc)

    def _record_target_head_component_gradients(
        self,
        components: tuple[tuple[str, torch.Tensor], ...],
    ) -> None:
        collector = self._target_head_diagnostics
        if collector is None:
            return
        try:
            collector.observe_component_gradients(components, self.model)
        except Exception as exc:
            self._disable_target_head_diagnostics(collector, exc)

    def _snapshot_encoder_layer_parameters(self) -> None:
        collector = self._target_head_diagnostics
        if collector is None:
            return
        try:
            collector.snapshot_encoder_parameters(self.model)
        except Exception as exc:
            self._disable_target_head_diagnostics(collector, exc)

    def _record_encoder_layer_parameter_updates(
        self,
        optimizer_update_applied: bool,
    ) -> None:
        collector = self._target_head_diagnostics
        if collector is None:
            return
        try:
            collector.observe_encoder_parameter_updates(
                self.model,
                optimizer_update_applied=optimizer_update_applied,
            )
        except Exception as exc:
            self._disable_target_head_diagnostics(collector, exc)

    def collect_target_head_diagnostics(
        self,
        features: Callable[[], Iterable[torch.Tensor]],
        *,
        epoch: int,
        global_step: int,
        expected_rows: int,
    ) -> None:
        collector = self._target_head_diagnostics
        if collector is None:
            return
        try:
            collector.observe_post_update(
                self.model,
                features,
                device=self.device,
                autocast=self._autocast,
                epoch=epoch,
                global_step=global_step,
                expected_rows=expected_rows,
            )
        except Exception as exc:
            self._disable_target_head_diagnostics(collector, exc)

    def target_head_diagnostics_artifact(
        self,
        *,
        job_id: str,
        attempt: int,
        attempt_id: str,
        input_revision: int,
        manifest_sha256: str,
        job_config_sha256: str,
    ) -> JsonObject | None:
        collector = self._target_head_diagnostics
        if collector is None:
            return None
        return collector.artifact(
            job_id=job_id,
            attempt=attempt,
            attempt_id=attempt_id,
            input_revision=input_revision,
            manifest_sha256=manifest_sha256,
            model_definition_sha256=self.model_definition_sha256,
            job_config_sha256=job_config_sha256,
            completed_epochs=self.state.global_epoch,
        )

    @staticmethod
    def _disable_target_head_diagnostics(
        collector: TargetHeadDiagnosticsCollector,
        exc: Exception,
    ) -> None:
        if collector.disable():
            print(
                "диагностика выходной головки отключена: "
                + type(exc).__name__,
                file=sys.stderr,
                flush=True,
            )

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
            selection_decision = self.selection_state.update(selection_score)
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

    def _sample_gradient_interactions(self, samples: int) -> bool:
        interval = self.train_config.diagnostics.gradient_sample_every_steps
        return (
            interval is not None
            and samples < self.train_config.diagnostics.gradient_max_batches
            and (self.state.train_step + 1) % interval == 0
        )

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
            "model_definition_sha256": self.model_definition_sha256,
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
                # Поле сохранено неактивным, чтобы сохранить текущий надёжный
                # формат восстановления. Телеметрия больше не восстанавливается
                # в состояние обучения.
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
            "target_head_diagnostics": (
                None
                if self._target_head_diagnostics is None
                else self._target_head_diagnostics.recovery_document()
            ),
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
            "model_definition_sha256",
            "selection_state",
            "selection",
            "payload_shuffle_generator_state",
            "rng",
            "training_complete",
            "target_head_diagnostics",
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
        if payload["model_definition_sha256"] != self.model_definition_sha256:
            raise ValueError("training recovery model definition differs")
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
        diagnostics_state = payload["target_head_diagnostics"]
        if self._target_head_diagnostics is None:
            if diagnostics_state is not None:
                raise ValueError("training recovery diagnostics state is invalid")
        else:
            if diagnostics_state is None:
                raise ValueError("training recovery diagnostics state is missing")
            self._target_head_diagnostics.load_recovery_document(diagnostics_state)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            if features.size(0) == 0:
                return public_predictions(
                    self.model(features.to(self.device)),
                    self.model_contract,
                )

            predictions = None
            for offset in range(0, features.size(0), self.batch_size):
                batch_features = features[
                    offset:offset + self.batch_size
                ].to(self.device)
                batch_predictions = public_predictions(
                    self.model(batch_features),
                    self.model_contract,
                )
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

    def load_payload(self, checkpoint: dict[str, object]) -> None:
        self.model.load_state_dict(
            _tensor_state_dict(checkpoint.get("state_dict"), "checkpoint state")
        )

    def save(self, model_name: str, *, metadata: JsonObject) -> None:
        if self.selection is not None and self.best_state_dict is not None:
            self._restore_best_state_dict()

        save_model(
            model_name,
            self.model,
            metadata=metadata,
        )


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
