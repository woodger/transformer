import sys

from app.service.application.services import attempt_executor as _implementation

sys.modules[__name__] = _implementation

