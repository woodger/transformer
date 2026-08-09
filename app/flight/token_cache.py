import sys

from app.service.adapters.outbound.postgres import token_cache as _implementation

sys.modules[__name__] = _implementation

