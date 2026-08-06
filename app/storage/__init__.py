"""Lazy compatibility exports for worker checkpoint storage."""

from importlib import import_module

_EXPORTS = {
    "CHECKPOINT_FORMAT": (
        "app.worker.runtime.checkpoints.checkpoint",
        "CHECKPOINT_FORMAT",
    ),
    "MODELS_DIR": ("app.worker.runtime.checkpoints.checkpoint", "MODELS_DIR"),
    "load_checkpoint": (
        "app.worker.runtime.checkpoints.checkpoint",
        "load_checkpoint",
    ),
    "load_checkpoint_metadata": (
        "app.worker.runtime.checkpoints.checkpoint",
        "load_checkpoint_metadata",
    ),
    "model_path": ("app.worker.runtime.checkpoints.checkpoint", "model_path"),
    "save_checkpoint": (
        "app.worker.runtime.checkpoints.checkpoint",
        "save_checkpoint",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
