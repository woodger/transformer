import sys

from app.service.adapters.inbound.flight import job_actions as _implementation

sys.modules[__name__] = _implementation

