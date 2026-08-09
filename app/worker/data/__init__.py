"""Lazy compatibility exports for the worker-owned data package."""

from importlib import import_module

_EXPORTS = {
    "empty_predictions_table": ("app.worker.data.arrow", "empty_predictions_table"),
    "iter_framed_arrow": ("app.worker.data.arrow", "iter_framed_arrow"),
    "predictions_to_table": ("app.worker.data.arrow", "predictions_to_table"),
    "read_arrow": ("app.worker.data.arrow", "read_arrow"),
    "read_source_arrow": ("app.worker.data.arrow", "read_source_arrow"),
    "reshape_source": ("app.worker.data.tensors", "reshape_source"),
    "table_to_source_tensor": ("app.worker.data.arrow", "table_to_source_tensor"),
    "table_to_tensors": ("app.worker.data.arrow", "table_to_tensors"),
    "validate_checkpoint_feature_dim": (
        "app.worker.data.tensors",
        "validate_checkpoint_feature_dim",
    ),
    "validate_feature_dim": ("app.worker.data.tensors", "validate_feature_dim"),
    "validate_target_dim": ("app.worker.data.tensors", "validate_target_dim"),
    "write_arrow": ("app.worker.data.arrow", "write_arrow"),
    "write_framed_arrow": ("app.worker.data.arrow", "write_framed_arrow"),
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
