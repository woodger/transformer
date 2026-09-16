from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import (
    CUDA_TORCH_INTEROP_THREADS_ENV,
    CUDA_TORCH_INTRAOP_THREADS_ENV,
)
from app.worker.runtime.thread_budget import (
    configure_cuda_torch_thread_budget,
)


@dataclass
class _TorchRuntime:
    intraop_threads: int | None = None
    interop_threads: int | None = None

    def set_num_threads(self, threads: int) -> None:
        self.intraop_threads = threads

    def set_num_interop_threads(self, threads: int) -> None:
        self.interop_threads = threads


def test_cuda_worker_applies_complete_torch_thread_budget() -> None:
    runtime = _TorchRuntime()

    applied = configure_cuda_torch_thread_budget(
        {
            CUDA_TORCH_INTRAOP_THREADS_ENV: "8",
            CUDA_TORCH_INTEROP_THREADS_ENV: "1",
        },
        runtime=runtime,
    )

    assert applied == (8, 1)
    assert runtime == _TorchRuntime(
        intraop_threads=8,
        interop_threads=1,
    )


def test_worker_without_cuda_thread_budget_keeps_runtime_defaults() -> None:
    runtime = _TorchRuntime()

    assert configure_cuda_torch_thread_budget({}, runtime=runtime) is None
    assert runtime == _TorchRuntime()


@pytest.mark.parametrize(
    "environment",
    [
        {CUDA_TORCH_INTRAOP_THREADS_ENV: "8"},
        {CUDA_TORCH_INTEROP_THREADS_ENV: "1"},
        {
            CUDA_TORCH_INTRAOP_THREADS_ENV: "0",
            CUDA_TORCH_INTEROP_THREADS_ENV: "1",
        },
        {
            CUDA_TORCH_INTRAOP_THREADS_ENV: "eight",
            CUDA_TORCH_INTEROP_THREADS_ENV: "1",
        },
    ],
)
def test_worker_rejects_invalid_cuda_thread_budget(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        configure_cuda_torch_thread_budget(
            environment,
            runtime=_TorchRuntime(),
        )
