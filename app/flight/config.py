import sys

from app.service.bootstrap import config as _implementation

sys.modules[__name__] = _implementation

