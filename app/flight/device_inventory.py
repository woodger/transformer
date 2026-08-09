import sys

from app.service.adapters.outbound.worker_probe import (
    device_inventory as _implementation,
)

sys.modules[__name__] = _implementation

