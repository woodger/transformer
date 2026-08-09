import sys

from app.worker.training import loss_scheduler as _implementation

sys.modules[__name__] = _implementation
