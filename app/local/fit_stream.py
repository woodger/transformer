import sys
from collections.abc import Callable
from typing import Protocol, cast

import torch

from app.config import DEFAULT_MAX_FRAME_BYTES
from app.contracts.semantic.v1 import ModelContract
from app.contracts.worker.v12.config import ModelConfig
from app.local.semantic import (
    load_model_contract,
    local_checkpoint_metadata,
    stream_manifest_sha256,
)
from app.worker.data.arrow import iter_framed_arrow, table_to_tensors
from app.worker.data.tensors import (
    TrainingBatch,
    reshape_source,
    validate_feature_dim,
    validate_target_dim,
)
from app.worker.telemetry import format_epoch_console_line
from app.worker.telemetry.epoch import ObservedTrainingEpoch
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import SelectionPayload, Trainer

ModelBuilder = Callable[
    [object, torch.Tensor, torch.Tensor | None, torch.device, ModelContract],
    torch.nn.Module,
]
TrainerBuilder = Callable[..., Trainer]


class FitStreamArguments(Protocol):
    model_name: str
    model_contract: str


def run(
    args: FitStreamArguments,
    device: torch.device,
    build_model_fn: ModelBuilder = build_model,
    build_trainer_fn: TrainerBuilder = build_trainer,
) -> None:
    contract = load_model_contract(args.model_contract)
    model_config = ModelConfig.from_manifest(contract.model_config)
    model = None
    trainer = None
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

        raw_batch = table_to_tensors(table, contract.target_contract)
        batch = TrainingBatch(
            features=reshape_source(raw_batch.features, model_config.seq_len),
            targets=raw_batch.targets,
        )
        validate_feature_dim(batch.features, model_config.feature_dim)
        validate_target_dim(batch.targets, contract.target_width)

        if model is None:
            model = build_model_fn(
                model_config,
                batch.features,
                batch.targets,
                device,
                contract,
            )
            trainer = build_trainer_fn(
                args,
                model,
                device,
                model_config,
                model_contract=contract,
                initialization={"kind": "random"},
            )
            print(trainer.config_line())
            print("features:", batch.features.shape, "targets:", batch.targets.shape)

        if trainer is None:
            raise AssertionError("streaming trainer was not initialized")
        active_trainer = trainer
        frame = received_frames

        def on_epoch(
            epoch: int,
            metrics: ObservedTrainingEpoch,
            monitor: SelectionPayload,
            *,
            trainer_for_frame: Trainer = active_trainer,
            frame_number: int = frame,
        ) -> None:
            nonlocal trained_epochs
            trained_epochs += 1
            print(format_epoch_console_line(
                metrics,
                frame=frame_number,
                epoch=epoch + 1,
                selection_score=monitor["selection_score"],
                checkpoint_best=monitor["checkpoint_best"],
                should_stop=monitor["should_stop"],
                best_selection_score=monitor["best_selection_score"],
            ))
            trainer_for_frame.record_metrics(
                metrics,
                mode="fit-stream",
                frame=frame_number,
                epoch=epoch + 1,
                selectionScore=monitor["selection_score"],
                checkpointBest=monitor["checkpoint_best"],
                shouldStop=monitor["should_stop"],
                bestSelectionScore=monitor["best_selection_score"],
            )

        active_trainer.fit_epochs(batch, on_epoch=on_epoch, frame=frame)
        trained_frames += 1

    if trainer is None:
        raise ValueError("No non-empty frames received on stdin")

    digest = stream_manifest_sha256(
        contract,
        received_frames=received_frames,
        trained_frames=trained_frames,
    )
    trainer.save(
        args.model_name,
        metadata=local_checkpoint_metadata(
            trainer,
            contract,
            source_identity="stdin",
            manifest_sha256=digest,
        ),
    )
    print(
        f"Model saved after {trained_frames} trained frame(s), "
        f"{trained_epochs} epoch(s) "
        f"from {received_frames} received frame(s)"
    )


def _max_frame_bytes(args: object) -> int:
    value = cast(
        object,
        getattr(args, "max_frame_bytes", DEFAULT_MAX_FRAME_BYTES),
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_frame_bytes must be an integer")
    return value


__all__ = ["FitStreamArguments", "run"]
