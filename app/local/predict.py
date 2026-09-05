import os
from collections.abc import Callable
from typing import Protocol, cast

import torch

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v1 import ModelContract
from app.contracts.worker.v12.config import ModelConfig
from app.local.semantic import checkpoint_model_contract
from app.worker.checkpoints.model import load_checkpoint
from app.worker.data.arrow import read_source_arrow, write_arrow
from app.worker.data.tensors import reshape_source, validate_checkpoint_feature_dim
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import Trainer

ModelBuilder = Callable[
    [object, torch.Tensor, torch.Tensor | None, torch.device, ModelContract],
    torch.nn.Module,
]
TrainerBuilder = Callable[..., Trainer]


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

    checkpoint = load_checkpoint(args.model_name, device)
    metadata = cast(JsonObject, checkpoint["metadata"])
    contract = checkpoint_model_contract(metadata)
    model_config = ModelConfig.from_manifest(contract.model_config)
    features_cpu = read_source_arrow(args.data)
    print("features:", features_cpu.shape)
    if features_cpu.shape[0] == 0:
        write_arrow(
            args.preds_path,
            torch.empty((0, contract.target_width), dtype=torch.float32),
            args.pred_col,
            target_contract=contract.target_contract,
        )
        print("Predictions saved")
        return
    features_cpu = reshape_source(features_cpu, model_config.seq_len)
    validate_checkpoint_feature_dim(features_cpu, model_config.feature_dim)

    model = build_model_fn(model_config, features_cpu, None, device, contract)
    trainer = build_trainer_fn(
        args,
        model,
        device,
        model_config,
        model_contract=contract,
    )
    trainer.load_payload(checkpoint)
    predictions = trainer.predict(features_cpu)
    write_arrow(
        args.preds_path,
        predictions,
        args.pred_col,
        expected_rows=features_cpu.shape[0],
        target_contract=contract.target_contract,
    )
    print("Predictions saved")


__all__ = ["ModelBuilder", "PredictArguments", "TrainerBuilder", "run"]
