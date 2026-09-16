from __future__ import annotations

from dataclasses import dataclass

import pytest

from app import config as settings
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

    applied = configure_cuda_torch_thread_budget(runtime=runtime)

    assert applied == (
        settings.CUDA_TORCH_INTRAOP_THREADS,
        settings.CUDA_TORCH_INTEROP_THREADS,
    )
    assert runtime == _TorchRuntime(
        intraop_threads=8,
        interop_threads=1,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CUDA_TORCH_INTRAOP_THREADS", 0),
        ("CUDA_TORCH_INTEROP_THREADS", -1),
        ("CUDA_TORCH_INTRAOP_THREADS", "eight"),
    ],
)
def test_worker_rejects_invalid_code_configured_thread_budget(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
) -> None:
    monkeypatch.setattr(settings, name, value)

    with pytest.raises(ValueError):
        configure_cuda_torch_thread_budget(runtime=_TorchRuntime())
