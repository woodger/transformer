import sys

from app.service.adapters.outbound.postgres import config as _implementation

sys.modules[__name__] = _implementation
