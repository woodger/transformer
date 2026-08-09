import sys

from app.service.adapters.outbound.postgres import ledger_inputs as _implementation

sys.modules[__name__] = _implementation

