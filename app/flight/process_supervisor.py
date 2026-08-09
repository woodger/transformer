import sys

from app.service.adapters.outbound.worker_process import (
    process_supervisor as _implementation,
)

sys.modules[__name__] = _implementation

