from collections.abc import Callable
from typing import Protocol

import torch

from app.contracts.semantic.v1 import ModelContract
from app.contracts.worker.v12.config import ModelConfig
from app.local.semantic import (
    file_sha256,
    load_model_contract,
    local_checkpoint_metadata,
    normalized_source_identity,
)
from app.worker.data.arrow import read_arrow
from app.worker.data.tensors import TrainingBatch, reshape_source
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import Trainer

ModelBuilder = Callable[
    [object, torch.Tensor, torch.Tensor | None, torch.device, ModelContract],
    torch.nn.Module,
]
TrainerBuilder = Callable[..., Trainer]


class FitArguments(Protocol):
    data: str | None
    model_name: str
    model_contract: str


def run(
    args: FitArguments,
    device: torch.device,
    build_model_fn: ModelBuilder = build_model,
    build_trainer_fn: TrainerBuilder = build_trainer,
) -> None:
    data_path = args.data
    if data_path is None:
        raise ValueError("data path is required for fit")
    contract = load_model_contract(args.model_contract)
    model_config = ModelConfig.from_manifest(contract.model_config)
    batch = read_arrow(data_path, contract.target_contract)
    print("features:", batch.features.shape, "targets:", batch.targets.shape)
    if batch.features.shape[0] == 0:
        raise ValueError("Training input contains no rows")
    batch = TrainingBatch(
        features=reshape_source(batch.features, model_config.seq_len),
        targets=batch.targets,
    )
    if batch.features.shape[2] != model_config.feature_dim:
        raise ValueError("input feature width differs from model contract")

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
    digest = file_sha256(data_path)
    trainer.fit(
        batch,
        args.model_name,
        metadata=lambda: local_checkpoint_metadata(
            trainer,
            contract,
            source_identity=normalized_source_identity(data_path),
            manifest_sha256=digest,
        ),
    )


__all__ = ["FitArguments", "ModelBuilder", "TrainerBuilder", "run"]
