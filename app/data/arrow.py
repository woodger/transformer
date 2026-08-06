import sys

from app.worker.data import arrow as _implementation

sys.modules[__name__] = _implementation
