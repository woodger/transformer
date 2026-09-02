from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

import torch

from app.contracts.worker.v10.objective import ObjectiveConfig
from app.worker.data.arrow import read_arrow
from app.worker.data.tensors import TrainingBatch, reshape_source
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.run_config import (
    model_config_from_args,
    objective_config_from_args,
)
from app.worker.training.trainer import Trainer

ModelBuilder = Callable[
    [object, torch.Tensor, torch.Tensor | None, torch.device, ObjectiveConfig],
    torch.nn.Module,
]
TrainerBuilder = Callable[..., Trainer]


class FitArguments(Protocol):
    data: str | None
    model_name: str


def run(
    args: FitArguments,
    device: torch.device,
    build_model_fn: ModelBuilder = build_model,
    build_trainer_fn: TrainerBuilder = build_trainer,
) -> None:
    if args.data is None:
        raise ValueError("data path is required for fit")
    model_config = model_config_from_args(args)
    objective = objective_config_from_args(args)
    batch = read_arrow(args.data, objective.targets)
    print("features:", batch.features.shape, "targets:", batch.targets.shape)
    if batch.features.shape[0] == 0:
        raise ValueError("Training input contains no rows")
    batch = TrainingBatch(
        features=reshape_source(batch.features, model_config.seq_len),
        targets=batch.targets,
    )
    model_config = replace(
        model_config,
        feature_dim=batch.features.shape[2],
        out_dim=objective.target_width,
    )

    model = build_model_fn(
        model_config,
        batch.features,
        batch.targets,
        device,
        objective,
    )
    trainer = build_trainer_fn(
        args,
        model,
        device,
        model_config,
        objective=objective,
    )
    trainer.fit(batch, args.model_name)
