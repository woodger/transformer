import os
from collections.abc import Callable
from typing import Protocol

import torch

from app.contracts.worker.v4.config import ModelConfig
from app.worker.checkpoints.model import load_checkpoint_metadata
from app.worker.data.arrow import read_source_arrow, write_arrow
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


class PredictArguments(Protocol):
    data: str | None
    model_name: str
    pred_col: str
    preds_path: str


def run(
    args: PredictArguments,
    device: torch.device,
    build_model_fn: ModelBuilder = build_model,
    build_trainer_fn: TrainerBuilder = build_trainer,
) -> None:
    if args.data is None:
        raise ValueError("data path is required for predict")
    if os.path.realpath(args.data) == os.path.realpath(args.preds_path):
        raise ValueError("prediction output path must differ from input data path")

    metadata = load_checkpoint_metadata(args.model_name, device)
    model_config = model_config_from_args(
        args,
        checkpoint_config=metadata.get("model_config"),
        require_seq_len=True,
    )

    features_cpu = read_source_arrow(args.data)
    print("features:", features_cpu.shape)
    if features_cpu.shape[0] == 0:
        write_arrow(
            args.preds_path,
            torch.empty((0, model_config.out_dim), dtype=torch.float32),
            args.pred_col,
        )
        print("Predictions saved")
        return
    features_cpu = reshape_source(features_cpu, model_config.seq_len)
    validate_checkpoint_feature_dim(features_cpu, model_config.feature_dim)

    model = build_model_fn(model_config, features_cpu, None, device)
    trainer = build_trainer_fn(args, model, device, model_config)
    trainer.load(args.model_name)
    predictions = trainer.predict(features_cpu)
    write_arrow(
        args.preds_path,
        predictions,
        args.pred_col,
        expected_rows=features_cpu.shape[0],
    )
    print("Predictions saved")
