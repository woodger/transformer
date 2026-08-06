import sys

from app.service.application.commands import jobs as _implementation

sys.modules[__name__] = _implementation
