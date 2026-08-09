import sys

from app.service.adapters.inbound.flight import upload_session as _implementation

sys.modules[__name__] = _implementation

