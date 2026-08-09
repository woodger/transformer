import sys

from app.service.bootstrap import maintenance as _implementation

sys.modules[__name__] = _implementation
