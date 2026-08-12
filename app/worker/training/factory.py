from collections.abc import Mapping
from typing import cast

import torch

from app.worker.model.transformer import TransformerModel
from app.worker.training.run_config import (
    ModelConfig,
    TrainConfig,
    model_config_from_args,
    train_config_from_args,
)
from app.worker.training.trainer import Trainer
from app.worker.utils import resolve_metrics_path


def build_model(
    args_or_config: object,
    features_cpu: torch.Tensor,
    targets_cpu: torch.Tensor | None,
    device: torch.device,
) -> TransformerModel:
    model_config = _coerce_model_config(args_or_config)
    feat_dim = features_cpu.shape[2]
    out_dim = (
        targets_cpu.shape[1]
        if targets_cpu is not None
        else model_config.out_dim
    )

    return TransformerModel(
        input_dim=feat_dim,
        seq_len=model_config.seq_len,
        hidden_dim=model_config.hidden,
        layers=model_config.layers,
        dropout=model_config.dropout,
        out_dim=out_dim,
        nhead=model_config.nhead,
        context_mode=model_config.context_mode,
    ).to(device)


def build_trainer(
    args_or_config: object,
    model: torch.nn.Module,
    device: torch.device,
    model_config: ModelConfig | None = None,
    data_contract: Mapping[str, object] | None = None,
) -> Trainer:
    train_config = _coerce_train_config(args_or_config)
    if model_config is None:
        model_config = _coerce_model_config(args_or_config)

    metrics_name_value = cast(
        object,
        getattr(args_or_config, "metrics_name", None),
    )
    if metrics_name_value is not None and not isinstance(
        metrics_name_value,
        str,
    ):
        raise ValueError("metrics_name must be a string")
    metrics_name = metrics_name_value
    metrics_path = resolve_metrics_path(metrics_name)

    return Trainer(
        model=model,
        device=device,
        lr=train_config.lr,
        batch_size=train_config.batch_size,
        epochs=train_config.epochs,
        loss_stage=train_config.loss_stage,
        loss_schedule=train_config.loss_schedule,
        stage_size=train_config.stage_size,
        use_amp=train_config.use_amp,
        weight_decay=train_config.weight_decay,
        direct_loss_weights=train_config.direct_loss_weights,
        selection=train_config.selection,
        metrics_path=metrics_path,
        context_mode=model_config.context_mode,
        metrics_context={
            "hidden": model_config.hidden,
            "layers": model_config.layers,
            "seq_len": model_config.seq_len,
        },
        model_config=model_config,
        train_config=train_config,
        data_contract=(
            None if data_contract is None else dict(data_contract)
        ),
        seed=train_config.seed,
    )


def _coerce_model_config(value: object) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    return model_config_from_args(value)


def _coerce_train_config(value: object) -> TrainConfig:
    if isinstance(value, TrainConfig):
        return value
    return train_config_from_args(value)
