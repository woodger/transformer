import sys

from app.admin.bootstrap import auth_tokens as _implementation

sys.modules[__name__] = _implementation
