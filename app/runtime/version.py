import sys

from app import version as _implementation

sys.modules[__name__] = _implementation
