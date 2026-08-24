from __future__ import annotations

__version__ = "0.1.16"


def version_text(prog: str = "main.py") -> str:
    return f"{prog} {__version__}"


__all__ = ["__version__", "version_text"]
