import torch

from app.config import CONTEXT_MODE
from app.worker.model.transformer import TransformerModel
from app.worker.training.run_config import (
    ModelConfig,
    TrainConfig,
    model_config_from_args,
    train_config_from_args,
)
from app.worker.training.trainer import Trainer
from app.worker.utils import resolve_metrics_path


def build_model(args_or_config, X_cpu: torch.Tensor, Y_cpu: torch.Tensor | None, device):
    model_config = _coerce_model_config(args_or_config)
    feat_dim = X_cpu.shape[2]
    out_dim = Y_cpu.shape[1] if Y_cpu is not None else model_config.out_dim

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
    args_or_config,
    model,
    device,
    model_config: ModelConfig | None = None,
    data_contract: dict | None = None,
):
    train_config = _coerce_train_config(args_or_config)
    if model_config is None:
        model_config = _coerce_model_config(args_or_config)

    metrics_name = getattr(args_or_config, "metrics_name", None)
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
        context_mode=model_config.context_mode if model_config else CONTEXT_MODE,
        metrics_context={
            "hidden": model_config.hidden,
            "layers": model_config.layers,
            "seq_len": model_config.seq_len,
        } if model_config else None,
        model_config=model_config,
        train_config=train_config,
        data_contract=data_contract,
        seed=train_config.seed,
    )


def _coerce_model_config(value):
    if isinstance(value, ModelConfig):
        return value
    return model_config_from_args(value)


def _coerce_train_config(value):
    if isinstance(value, TrainConfig):
        return value
    return train_config_from_args(value)
