import sys

from app.service.adapters import observability as _implementation

sys.modules[__name__] = _implementation

