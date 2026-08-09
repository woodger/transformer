import sys

from app.worker.metrics import io as _implementation

sys.modules[__name__] = _implementation
