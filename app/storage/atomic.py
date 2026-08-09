import sys

from app.worker.runtime.checkpoints import atomic as _implementation

sys.modules[__name__] = _implementation
