import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Protocol, cast

import torch

from app.contracts.worker.v7.config import ModelConfig
from app.local.config import DEFAULT_MAX_FRAME_BYTES
from app.worker.checkpoints.model import load_checkpoint
from app.worker.data.arrow import (
    empty_predictions_table,
    iter_framed_arrow,
    predictions_to_table,
    table_to_source_tensor,
    write_framed_arrow,
)
from app.worker.data.tensors import (
    reshape_source,
    validate_checkpoint_feature_dim,
)
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.run_config import model_config_from_args
from app.worker.training.trainer import Trainer

ModelBuilder = Callable[
    [object, torch.Tensor, torch.Tensor | None, torch.device],
    torch.nn.Module,
]
TrainerBuilder = Callable[
    [object, torch.nn.Module, torch.device, ModelConfig | None],
    Trainer,
]


class PredictStreamArguments(Protocol):
    model_name: str
    pred_col: str


def run(
    args: PredictStreamArguments,
    device: torch.device,
    build_model_fn: ModelBuilder = build_model,
    build_trainer_fn: TrainerBuilder = build_trainer,
) -> None:
    with redirect_stdout(sys.stderr):
        checkpoint: dict[str, object] | None = load_checkpoint(
            args.model_name,
            device,
        )
    model_config = model_config_from_args(
        args,
        checkpoint_config=checkpoint.get("model_config"),
        require_seq_len=True,
    )
    model = None
    trainer = None
    received_frames = 0
    predicted_frames = 0

    for table in iter_framed_arrow(
        sys.stdin.buffer,
        max_frame_bytes=_max_frame_bytes(args),
    ):
        received_frames += 1
        if table.num_rows == 0:
            write_framed_arrow(
                sys.stdout.buffer,
                empty_predictions_table(args.pred_col),
            )
            print(
                f"frame {received_frames}, emitted empty predictions",
                file=sys.stderr,
            )
            continue

        features_cpu = table_to_source_tensor(table)
        features_cpu = reshape_source(features_cpu, model_config.seq_len)
        validate_checkpoint_feature_dim(
            features_cpu,
            model_config.feature_dim,
        )

        if model is None:
            with redirect_stdout(sys.stderr):
                model = build_model_fn(
                    model_config,
                    features_cpu,
                    None,
                    device,
                )
                trainer = build_trainer_fn(args, model, device, model_config)
                if checkpoint is None:
                    raise AssertionError(
                        "prediction checkpoint was already consumed"
                    )
                trainer.load_payload(checkpoint)
                checkpoint = None
            print("features:", features_cpu.shape, file=sys.stderr)
            print("Model loaded", file=sys.stderr)

        with redirect_stdout(sys.stderr):
            if trainer is None:
                raise AssertionError("prediction trainer was not initialized")
            predictions = trainer.predict(features_cpu)
        write_framed_arrow(
            sys.stdout.buffer,
            predictions_to_table(
                predictions,
                args.pred_col,
                expected_rows=features_cpu.shape[0],
            ),
        )
        predicted_frames += 1
        print(
            f"frame {received_frames}, predicted {predictions.shape[0]} row(s)",
            file=sys.stderr,
        )

    print(
        f"Predicted {predicted_frames} non-empty frame(s) "
        f"from {received_frames} received frame(s)",
        file=sys.stderr,
    )


def _max_frame_bytes(args: object) -> int:
    value = cast(
        object,
        getattr(args, "max_frame_bytes", DEFAULT_MAX_FRAME_BYTES),
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_frame_bytes must be an integer")
    return value
