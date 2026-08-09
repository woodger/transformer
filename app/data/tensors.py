import sys

from app.worker.data import tensors as _implementation

sys.modules[__name__] = _implementation
