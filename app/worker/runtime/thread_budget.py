from __future__ import annotations

from typing import Protocol, cast

from app import config as settings


class _TorchThreadRuntime(Protocol):
    def set_num_threads(self, threads: int) -> None: ...

    def set_num_interop_threads(self, threads: int) -> None: ...


def configure_cuda_torch_thread_budget(
    *,
    runtime: _TorchThreadRuntime | None = None,
) -> tuple[int, int]:
    """Apply the code-configured thread budget in a fresh CUDA worker."""

    intraop_threads = _positive_integer(
        settings.CUDA_TORCH_INTRAOP_THREADS,
        "CUDA_TORCH_INTRAOP_THREADS",
    )
    interop_threads = _positive_integer(
        settings.CUDA_TORCH_INTEROP_THREADS,
        "CUDA_TORCH_INTEROP_THREADS",
    )
    if runtime is None:
        import torch

        runtime = cast(_TorchThreadRuntime, torch)
    runtime.set_num_threads(intraop_threads)
    runtime.set_num_interop_threads(interop_threads)
    return intraop_threads, interop_threads


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


__all__ = ["configure_cuda_torch_thread_budget"]
