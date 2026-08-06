from __future__ import annotations

import platform

__version__ = "0.1.7"


def version_text(prog: str = "main.py") -> str:
    import torch

    cuda_status = "available" if torch.cuda.is_available() else "unavailable"
    return "\n".join([
        f"{prog} {__version__}",
        f"python {platform.python_version()}",
        f"torch {torch.__version__}",
        f"cuda {cuda_status}",
    ])


__all__ = ["__version__", "version_text"]
