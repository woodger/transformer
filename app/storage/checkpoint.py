import sys

from app.worker.runtime.checkpoints import checkpoint as _implementation

sys.modules[__name__] = _implementation
