import sys

from app.worker.training import run_config as _implementation

sys.modules[__name__] = _implementation
