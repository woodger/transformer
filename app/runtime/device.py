import sys

from app.worker.runtime import device as _implementation

sys.modules[__name__] = _implementation
