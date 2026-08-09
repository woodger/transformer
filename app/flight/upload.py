import sys

from app.service.adapters.inbound.flight import upload as _implementation

sys.modules[__name__] = _implementation

