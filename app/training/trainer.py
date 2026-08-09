import sys

from app.worker.training import trainer as _implementation

sys.modules[__name__] = _implementation
