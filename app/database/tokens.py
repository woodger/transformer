import sys

from app.service.adapters.outbound.postgres import tokens as _implementation

sys.modules[__name__] = _implementation
