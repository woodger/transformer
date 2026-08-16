from importlib import import_module
from types import ModuleType

_COMMAND_MODULES = frozenset({
    "fit",
    "fit_stream",
    "gmark",
    "plot_metrics",
    "predict",
    "predict_stream",
})

# Pyright cannot infer module exports materialized by ``__getattr__``.
__all__ = sorted(_COMMAND_MODULES)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> ModuleType:
    if name not in _COMMAND_MODULES:
        raise AttributeError(name)
    # Commands stay lazy so service and admin CLI bootstrap do not import
    # Torch before the selected command owns the ML runtime.
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
