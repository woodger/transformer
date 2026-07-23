import torch

from app.storage.checkpoint import MODELS_DIR, load_checkpoint, save_checkpoint
from app.storage.atomic import resolve_artifact_path


def save_model(model_name: str, model, model_config=None, train_config=None, extra=None):
    save_checkpoint(
        model_name,
        model,
        model_config=model_config,
        train_config=train_config,
        extra=extra,
    )


def load_model(model_name: str, model, device):
    checkpoint = load_checkpoint(model_name, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def resolve_metrics_path(metrics_name: str | None):
    if metrics_name is None:
        return None

    return resolve_artifact_path(
        metrics_name,
        MODELS_DIR,
        label="metrics path",
    )


def tree_stats(params):
    cpu_tensors = [p.detach().cpu().flatten() for p in params]
    flat = torch.cat(cpu_tensors)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "norm": float(torch.norm(flat)),
    }
