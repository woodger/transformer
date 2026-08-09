"""Lazy runtime exports; importing service metadata must not initialize Torch."""

from importlib import import_module

_EXPORTS = {
    "__version__": ("app.version", "__version__"),
    "get_device": ("app.worker.runtime.device", "get_device"),
    "version_text": ("app.version", "version_text"),
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
