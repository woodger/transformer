import sys

from app.worker.model import positional_encoding as _implementation

sys.modules[__name__] = _implementation
