import sys

from app.worker.training import early_stopping as _implementation

sys.modules[__name__] = _implementation
