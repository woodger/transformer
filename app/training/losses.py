import sys

from app.worker.training import losses as _implementation

sys.modules[__name__] = _implementation
