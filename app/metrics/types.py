import sys

from app.worker.metrics import types as _implementation

sys.modules[__name__] = _implementation
