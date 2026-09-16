from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol, cast

from app.config import (
    CUDA_TORCH_INTEROP_THREADS_ENV,
    CUDA_TORCH_INTRAOP_THREADS_ENV,
)


class _TorchThreadRuntime(Protocol):
    def set_num_threads(self, threads: int) -> None: ...

    def set_num_interop_threads(self, threads: int) -> None: ...


def configure_cuda_torch_thread_budget(
    environ: Mapping[str, str] | None = None,
    *,
    runtime: _TorchThreadRuntime | None = None,
) -> tuple[int, int] | None:
    """Apply the service-owned thread budget in a fresh CUDA worker."""

    values = os.environ if environ is None else environ
    intraop_value = values.get(CUDA_TORCH_INTRAOP_THREADS_ENV)
    interop_value = values.get(CUDA_TORCH_INTEROP_THREADS_ENV)
    if intraop_value is None and interop_value is None:
        return None
    if intraop_value is None or interop_value is None:
        raise ValueError("CUDA Torch thread budget is incomplete")

    intraop_threads = _positive_integer(
        intraop_value,
        CUDA_TORCH_INTRAOP_THREADS_ENV,
    )
    interop_threads = _positive_integer(
        interop_value,
        CUDA_TORCH_INTEROP_THREADS_ENV,
    )
    if runtime is None:
        import torch

        runtime = cast(_TorchThreadRuntime, torch)
    runtime.set_num_threads(intraop_threads)
    runtime.set_num_interop_threads(interop_threads)
    return intraop_threads, interop_threads


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


__all__ = ["configure_cuda_torch_thread_budget"]
