import sys

from app.service.adapters.outbound.postgres import models as _implementation

sys.modules[__name__] = _implementation
