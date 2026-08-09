import sys

from app.service.adapters.outbound.worker_process import plan as _implementation

sys.modules[__name__] = _implementation

