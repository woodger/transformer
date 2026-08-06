import sys

from app.admin.bootstrap import db_migrations as _implementation

sys.modules[__name__] = _implementation
