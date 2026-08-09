import sys

from app.service.adapters.inbound.flight import contract as _implementation

sys.modules[__name__] = _implementation

