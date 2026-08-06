import sys

from app.worker.model import context as _implementation

sys.modules[__name__] = _implementation
