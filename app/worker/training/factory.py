from collections.abc import Mapping

import torch

from app.contracts.semantic.v6 import ModelContract
from app.worker.model.transformer import TransformerModel
from app.worker.training.run_config import (
    ModelConfig,
    TrainConfig,
    model_config_from_args,
    train_config_from_args,
)
from app.worker.training.trainer import Trainer


def build_model(
    args_or_config: object,
    features_cpu: torch.Tensor,
    targets_cpu: torch.Tensor | None,
    device: torch.device,
    model_contract: ModelContract,
) -> TransformerModel:
    model_config = _coerce_model_config(args_or_config)
    feat_dim = features_cpu.shape[2]
    if model_config.to_tuning() != model_contract.model_tuning:
        raise ValueError("model configuration differs from model tuning")
    if (
        targets_cpu is not None
        and targets_cpu.shape[1] != model_contract.target_width
    ):
        raise ValueError("training target width differs from objective targets")

    return TransformerModel(
        input_dim=feat_dim,
        seq_len=model_config.seq_len,
        hidden_dim=model_config.hidden,
        layers=model_config.layers,
        dropout=model_config.dropout,
        model_contract=model_contract,
        nhead=model_config.nhead,
        context_mode=model_config.context_mode,
        normalization_order=model_config.normalization_order,
    ).to(device)


def build_trainer(
    args_or_config: object,
    model: torch.nn.Module,
    device: torch.device,
    model_config: ModelConfig | None = None,
    *,
    model_contract: ModelContract,
    model_definition_sha256: str,
    initialization: Mapping[str, object] | None = None,
) -> Trainer:
    train_config = _coerce_train_config(args_or_config)
    if model_config is None:
        model_config = _coerce_model_config(args_or_config)
    return Trainer(
        model=model,
        device=device,
        train_config=train_config,
        model_contract=model_contract,
        context_mode=model_config.context_mode,
        model_config=model_config,
        model_definition_sha256=model_definition_sha256,
        initialization=initialization,
    )


def _coerce_model_config(value: object) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    return model_config_from_args(value)


def _coerce_train_config(value: object) -> TrainConfig:
    if isinstance(value, TrainConfig):
        return value
    return train_config_from_args(value)
