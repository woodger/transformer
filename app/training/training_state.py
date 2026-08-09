import sys

from app.worker.training import training_state as _implementation

sys.modules[__name__] = _implementation
