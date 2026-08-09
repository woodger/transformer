import sys

from app.service.adapters.outbound.artifact_storage import spool as _implementation

sys.modules[__name__] = _implementation

