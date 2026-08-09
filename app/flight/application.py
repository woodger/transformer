import sys

from app.service.bootstrap import application as _implementation

sys.modules[__name__] = _implementation

