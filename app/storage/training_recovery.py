import sys

from app.worker.runtime.checkpoints import training_recovery as _implementation

sys.modules[__name__] = _implementation
