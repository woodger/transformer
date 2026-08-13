import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import torch

from app.contracts.json_types import JsonObject
from app.contracts.worker.v3.config import ModelConfig
from app.contracts.worker.v3.objective import objective_config_sha256
from app.local.config import DEFAULT_MAX_FRAME_BYTES
from app.worker.checkpoints.recovery import (
    load_training_recovery,
    save_training_recovery,
)
from app.worker.data.arrow import (
    iter_framed_arrow,
    read_arrow,
    table_to_tensors,
)
from app.worker.data.tensors import (
    TrainingBatch,
    reshape_source,
    validate_feature_dim,
    validate_target_dim,
)
from app.worker.metrics import TrainMetrics
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.run_config import model_config_from_args
from app.worker.training.trainer import SelectionPayload, Trainer

ModelBuilder = Callable[
    [object, torch.Tensor, torch.Tensor | None, torch.device],
    torch.nn.Module,
]
TrainerBuilder = Callable[
    [object, torch.nn.Module, torch.device, ModelConfig | None],
    Trainer,
]


class FitStreamArguments(Protocol):
    model_name: str


@dataclass(frozen=True, slots=True)
class RecoveryArguments:
    checkpoint_dir: str
    events_path: str
    config_hash: str
    manifest_hash: str
    resume_checkpoint: str | None


def run(
    args: FitStreamArguments,
    device: torch.device,
    build_model_fn: ModelBuilder = build_model,
    build_trainer_fn: TrainerBuilder = build_trainer,
) -> None:
    input_spool_dir = _optional_string_argument(args, "input_spool_dir")
    input_frame_count = _optional_integer_argument(
        args,
        "input_frame_count",
    )
    if input_spool_dir is not None or input_frame_count is not None:
        return _run_spooled(
            args,
            device,
            build_model_fn,
            build_trainer_fn,
            input_spool_dir,
            input_frame_count,
        )

    model_config = model_config_from_args(args)
    model = None
    trainer = None
    expected_feat_dim = None
    expected_target_dim = None
    received_frames = 0
    trained_frames = 0
    trained_epochs = 0

    for table in iter_framed_arrow(
        sys.stdin.buffer,
        max_frame_bytes=_max_frame_bytes(args),
    ):
        received_frames += 1
        if table.num_rows == 0:
            print(f"frame {received_frames}, skipped empty payload")
            continue

        batch = table_to_tensors(table)
        batch = TrainingBatch(
            features=reshape_source(batch.features, model_config.seq_len),
            targets=batch.targets,
        )
        expected_feat_dim = validate_feature_dim(
            batch.features,
            expected_feat_dim,
        )
        expected_target_dim = validate_target_dim(
            batch.targets,
            expected_target_dim,
        )

        if model is None:
            model_config = replace(model_config, feature_dim=expected_feat_dim)
            model = build_model_fn(
                model_config,
                batch.features,
                batch.targets,
                device,
            )
            trainer = build_trainer_fn(args, model, device, model_config)
            _print_config_line(trainer)
            print(
                "features:",
                batch.features.shape,
                "targets:",
                batch.targets.shape,
            )

        if trainer is None:
            raise AssertionError("streaming trainer was not initialized")
        active_trainer = trainer
        frame = received_frames

        def on_epoch(
            epoch: int,
            metrics: TrainMetrics,
            monitor_payload: SelectionPayload,
            current_frame: int = frame,
            current_trainer: Trainer = active_trainer,
        ) -> None:
            nonlocal trained_epochs
            trained_epochs += 1
            print(metrics.console_line(
                frame=current_frame,
                epoch=epoch + 1,
                **_metrics_context(current_trainer),
                selection_score=monitor_payload["selection_score"],
                checkpoint_best=monitor_payload["checkpoint_best"],
                should_stop=monitor_payload["should_stop"],
                best_selection_score=monitor_payload[
                    "best_selection_score"
                ],
            ))
            current_trainer.record_metrics(
                metrics,
                mode="fit-stream",
                frame=current_frame,
                epoch=epoch + 1,
                selection_score=monitor_payload["selection_score"],
                checkpoint_best=monitor_payload["checkpoint_best"],
                should_stop=monitor_payload["should_stop"],
                best_selection_score=monitor_payload[
                    "best_selection_score"
                ],
            )

        active_trainer.fit_epochs(
            batch,
            on_epoch=on_epoch,
            frame=received_frames,
        )
        trained_frames += 1

    if trainer is None:
        raise ValueError("No non-empty frames received on stdin")

    trainer.save(args.model_name)
    print(
        f"Model saved after {trained_frames} trained frame(s), "
        f"{trained_epochs} epoch(s) "
        f"from {received_frames} received frame(s)"
    )


def _run_spooled(
    args: FitStreamArguments,
    device: torch.device,
    build_model_fn: ModelBuilder,
    build_trainer_fn: TrainerBuilder,
    input_spool_dir: str | None,
    input_frame_count: int | None,
) -> None:
    if input_spool_dir is None or input_frame_count is None:
        raise ValueError(
            "input_spool_dir and input_frame_count must be provided together"
        )
    if input_frame_count < 0:
        raise ValueError("input_frame_count must be a non-negative integer")

    model_config = model_config_from_args(args)
    recovery = _recovery_arguments(args)
    model = None
    trainer = None
    expected_feat_dim = None
    expected_target_dim = None
    trained_inputs: list[Path] = []
    input_directory = Path(input_spool_dir)

    # Preflight one durable payload at a time so shape errors are reported
    # before the optimizer mutates model state.
    for ordinal in range(input_frame_count):
        frame = ordinal + 1
        path = input_directory / f"{ordinal}.arrow"
        batch = read_arrow(str(path))
        if batch.features.size(0) == 0:
            print(f"frame {frame}, skipped empty payload")
            del batch
            continue

        batch = TrainingBatch(
            features=reshape_source(batch.features, model_config.seq_len),
            targets=batch.targets,
        )
        expected_feat_dim = validate_feature_dim(
            batch.features,
            expected_feat_dim,
        )
        expected_target_dim = validate_target_dim(
            batch.targets,
            expected_target_dim,
        )

        if model is None:
            model_config = replace(model_config, feature_dim=expected_feat_dim)
            model = build_model_fn(
                model_config,
                batch.features,
                batch.targets,
                device,
            )
            trainer = build_trainer_fn(args, model, device, model_config)
            _print_config_line(trainer)
            print(
                "features:",
                batch.features.shape,
                "targets:",
                batch.targets.shape,
            )

        trained_inputs.append(path)
        del batch

    if trainer is None:
        raise ValueError("No non-empty frames received in input spool")

    if recovery is not None and recovery.resume_checkpoint is not None:
        train_config = trainer.train_config
        try:
            payload = load_training_recovery(
                recovery.resume_checkpoint,
                device,
                expected_config_hash=recovery.config_hash,
                expected_manifest_hash=recovery.manifest_hash,
                expected_objective_config_sha256=objective_config_sha256(
                    train_config
                ),
            )
            if payload["model_config"] != asdict(model_config):
                raise ValueError(
                    "training recovery model configuration does not match"
                )
            if payload["train_config"] != asdict(train_config):
                raise ValueError(
                    "training recovery train configuration does not match"
                )
            trainer.load_recovery_state_dict(
                _object_mapping(
                    payload["trainer_state"],
                    "training recovery trainer state",
                )
            )
        except Exception as exc:
            raise ValueError(
                "training recovery checkpoint could not be restored"
            ) from exc

    def payloads() -> Iterator[TrainingBatch]:
        for path in trained_inputs:
            batch = read_arrow(str(path))
            batch = TrainingBatch(
                features=reshape_source(batch.features, model_config.seq_len),
                targets=batch.targets,
            )
            validate_feature_dim(batch.features, expected_feat_dim)
            validate_target_dim(batch.targets, expected_target_dim)
            yield batch
            del batch

    trained_epochs = 0

    def on_epoch(
        epoch: int,
        metrics: TrainMetrics,
        monitor_payload: SelectionPayload,
    ) -> None:
        nonlocal trained_epochs
        trained_epochs += 1
        print(metrics.console_line(
            epoch=epoch + 1,
            **_metrics_context(trainer),
            selection_score=monitor_payload["selection_score"],
            checkpoint_best=monitor_payload["checkpoint_best"],
            should_stop=monitor_payload["should_stop"],
            best_selection_score=monitor_payload["best_selection_score"],
        ))
        trainer.record_metrics(
            metrics,
            mode="fit-stream",
            epoch=epoch + 1,
            selection_score=monitor_payload["selection_score"],
            checkpoint_best=monitor_payload["checkpoint_best"],
            should_stop=monitor_payload["should_stop"],
            best_selection_score=monitor_payload["best_selection_score"],
        )

    def on_epoch_committed(
        _epoch: int,
        _metrics: TrainMetrics,
        _monitor_payload: SelectionPayload,
        _training_complete: bool,
    ) -> None:
        if recovery is None:
            raise AssertionError("training recovery is not configured")
        generation = trainer.state.global_epoch
        checkpoint_path = os.path.join(
            recovery.checkpoint_dir,
            f"{generation}.pth",
        )
        event = save_training_recovery(
            checkpoint_path,
            trainer,
            generation=generation,
            config_hash=recovery.config_hash,
            manifest_hash=recovery.manifest_hash,
        )
        _append_recovery_event(recovery.events_path, event)

    if not _training_complete(trainer):
        if recovery is None:
            trainer.fit_payloads(payloads, on_epoch=on_epoch)
        else:
            trainer.fit_payloads_resumable(
                payloads,
                on_epoch=on_epoch,
                on_epoch_committed=on_epoch_committed,
            )
    trainer.save(args.model_name)
    print(
        f"Model saved after {len(trained_inputs)} trained frame(s), "
        f"{trained_epochs} epoch(s) from {input_frame_count} received frame(s)"
    )


def _recovery_arguments(args: object) -> RecoveryArguments | None:
    checkpoint_dir = _optional_string_argument(
        args,
        "recovery_checkpoint_dir",
    )
    events_path = _optional_string_argument(args, "recovery_events_out")
    config_hash = _optional_string_argument(args, "recovery_config_hash")
    manifest_hash = _optional_string_argument(
        args,
        "recovery_manifest_hash",
    )
    resume_checkpoint = _optional_string_argument(args, "resume_checkpoint")
    required = (checkpoint_dir, events_path, config_hash, manifest_hash)
    configured = [value is not None for value in required]
    if not any(configured):
        if resume_checkpoint is not None:
            raise ValueError(
                "resume_checkpoint requires training recovery outputs"
            )
        return None
    if not all(configured):
        raise ValueError(
            "training recovery arguments must be provided together"
        )
    if (
        checkpoint_dir is None
        or events_path is None
        or config_hash is None
        or manifest_hash is None
    ):
        raise AssertionError("training recovery arguments were not narrowed")
    return RecoveryArguments(
        checkpoint_dir=checkpoint_dir,
        events_path=events_path,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        resume_checkpoint=resume_checkpoint,
    )


def _append_recovery_event(path: str, event: JsonObject) -> None:
    destination = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    line = json.dumps(
        event,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    with open(destination, "ab", buffering=0) as target:
        target.write(line)
        os.fsync(target.fileno())


def _max_frame_bytes(args: object) -> int:
    value = cast(
        object,
        getattr(args, "max_frame_bytes", DEFAULT_MAX_FRAME_BYTES),
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_frame_bytes must be an integer")
    return value


def _optional_string_argument(args: object, name: str) -> str | None:
    value = cast(object, getattr(args, name, None))
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string")


def _optional_integer_argument(args: object, name: str) -> int | None:
    value = cast(object, getattr(args, name, None))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _print_config_line(trainer: object) -> None:
    value = cast(object, getattr(trainer, "config_line", None))
    if value is None:
        return
    if not callable(value):
        raise ValueError("trainer config_line must be callable")
    print(cast(Callable[[], object], value)())


def _training_complete(trainer: object) -> bool:
    return bool(cast(object, getattr(trainer, "training_complete", False)))


def _metrics_context(trainer: object) -> JsonObject:
    value = cast(object, getattr(trainer, "metrics_context", {}))
    if not isinstance(value, Mapping):
        raise ValueError("trainer metrics context must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("trainer metrics context keys must be strings")
    return cast(JsonObject, dict(mapping))


def _object_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} keys must be strings")
    return cast(Mapping[str, object], mapping)
