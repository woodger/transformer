import sys

from app.worker.model import transformer as _implementation

sys.modules[__name__] = _implementation
