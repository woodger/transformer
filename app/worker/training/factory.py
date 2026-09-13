from collections.abc import Mapping
from typing import cast

import torch

from app.contracts.semantic.v2 import ModelContract
from app.worker.model.transformer import TransformerModel
from app.worker.telemetry.paths import resolve_metrics_path
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
    if model_config.to_manifest() != model_contract.model_config:
        raise ValueError("model configuration differs from model contract")
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
    ).to(device)


def build_trainer(
    args_or_config: object,
    model: torch.nn.Module,
    device: torch.device,
    model_config: ModelConfig | None = None,
    data_contract: Mapping[str, object] | None = None,
    *,
    metrics_path: str | None = None,
    model_contract: ModelContract,
    initialization: Mapping[str, object] | None = None,
) -> Trainer:
    train_config = _coerce_train_config(args_or_config)
    if model_config is None:
        model_config = _coerce_model_config(args_or_config)
    if metrics_path is None:
        metrics_name_value = cast(
            object,
            getattr(args_or_config, "metrics_name", None),
        )
        if metrics_name_value is not None and not isinstance(
            metrics_name_value,
            str,
        ):
            raise ValueError("metrics_name must be a string")
        metrics_path = resolve_metrics_path(metrics_name_value)

    return Trainer(
        model=model,
        device=device,
        train_config=train_config,
        model_contract=model_contract,
        metrics_path=metrics_path,
        context_mode=model_config.context_mode,
        metrics_context={
            "hidden": model_config.hidden,
            "layers": model_config.layers,
            "seq_len": model_config.seq_len,
        },
        model_config=model_config,
        data_contract=(
            None if data_contract is None else dict(data_contract)
        ),
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
