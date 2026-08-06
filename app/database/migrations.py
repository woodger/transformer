import sys

from app.service.adapters.outbound.postgres import migrations as _implementation

sys.modules[__name__] = _implementation
