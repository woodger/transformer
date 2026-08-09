import sys

from app.worker.metrics import plot as _implementation

sys.modules[__name__] = _implementation
