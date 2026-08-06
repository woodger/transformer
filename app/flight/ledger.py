import sys

from app.service.adapters.outbound.postgres import ledger as _implementation

sys.modules[__name__] = _implementation

