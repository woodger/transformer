from collections.abc import Iterable, Mapping
from typing import cast

import torch

from app.contracts.worker.v3.config import ModelConfig, TrainConfig
from app.worker.runtime.checkpoints.atomic import resolve_artifact_path
from app.worker.runtime.checkpoints.checkpoint import (
    MODELS_DIR,
    load_checkpoint,
    save_checkpoint,
)


def save_model(
    model_name: str,
    model: torch.nn.Module,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    data_contract: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    save_checkpoint(
        model_name,
        model,
        model_config=model_config,
        train_config=train_config,
        data_contract=data_contract,
        extra=extra,
    )


def load_model(
    model_name: str,
    model: torch.nn.Module,
    device: str | torch.device,
) -> torch.nn.Module:
    checkpoint = load_checkpoint(model_name, device)
    state_value = checkpoint["state_dict"]
    if not isinstance(state_value, Mapping):
        raise ValueError("checkpoint model state must be an object")
    state = cast(Mapping[object, object], state_value)
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("checkpoint model state must contain tensors")
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state))
    return model


def resolve_metrics_path(metrics_name: str | None) -> str | None:
    if metrics_name is None:
        return None

    return resolve_artifact_path(
        metrics_name,
        MODELS_DIR,
        label="metrics path",
    )


def tree_stats(params: Iterable[torch.Tensor]) -> dict[str, float]:
    cpu_tensors = [parameter.detach().cpu().flatten() for parameter in params]
    flat = torch.cat(cpu_tensors)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "norm": float(flat.square().sum().sqrt()),
    }
