import os

from app.project import PROJECT_ROOT
from app.worker.checkpoints.atomic import resolve_artifact_path

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def resolve_metrics_path(metrics_name: str | None) -> str | None:
    if metrics_name is None:
        return None

    return resolve_artifact_path(
        metrics_name,
        MODELS_DIR,
        label="metrics path",
    )


__all__ = ["MODELS_DIR", "resolve_metrics_path"]
