import sys

from app.worker.training import factory as _implementation

sys.modules[__name__] = _implementation
