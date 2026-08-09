import sys

from app.worker import utils as _implementation

sys.modules[__name__] = _implementation
