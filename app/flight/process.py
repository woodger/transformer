import sys

from app.service.adapters.outbound.worker_process import process as _implementation

sys.modules[__name__] = _implementation

