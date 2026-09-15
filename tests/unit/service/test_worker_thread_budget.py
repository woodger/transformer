from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from app.config import (
    CUDA_TORCH_INTEROP_THREADS_ENV,
    CUDA_TORCH_INTRAOP_THREADS_ENV,
)
from app.contracts.worker.v13 import CONTRACT_VERSION
from app.service.adapters.outbound.worker.runner import (
    WorkerSubprocessError,
    WorkerSubprocessRunner,
)


class _Spool:
    def __init__(self, root: Path) -> None:
        self.root = root

    def attempt_stdout_path(self, *_args: object) -> str:
        return str(self.root / "stdout.log")

    def attempt_stderr_path(self, *_args: object) -> str:
        return str(self.root / "stderr.log")

    def ensure_parent(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _launch_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: str,
) -> dict[str, str]:
    captured: dict[str, str] = {}

    def reject_launch(*_args: object, **kwargs: object) -> None:
        environment = cast(dict[str, str], kwargs["env"])
        captured.update(environment)
        raise OSError("stop after environment capture")

    runner = WorkerSubprocessRunner(
        SimpleNamespace(
            subprocess_timeout_seconds=5.0,
            cancel_grace_seconds=1.0,
            cuda_torch_intraop_threads=8,
            cuda_torch_interop_threads=1,
        ),
        object(),
        _Spool(tmp_path),
        logger=object(),
        popen_factory=reject_launch,
        python_executable=sys.executable,
    )
    monkeypatch.setenv(CUDA_TORCH_INTRAOP_THREADS_ENV, "99")
    monkeypatch.setenv(CUDA_TORCH_INTEROP_THREADS_ENV, "99")
    job = SimpleNamespace(
        job_id="11111111-1111-4111-8111-111111111111",
        attempt=1,
        attempt_id="22222222-2222-4222-8222-222222222222",
        selected_device=device,
        assigned_device_id="GPU-a" if device == "cuda" else None,
    )
    plan = SimpleNamespace(
        protocol_version=CONTRACT_VERSION,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
    )

    with pytest.raises(WorkerSubprocessError):
        runner.run(
            job,
            plan,
            cancel=threading.Event(),
            force_stop=threading.Event(),
        )
    return captured


def test_cuda_worker_receives_configured_torch_thread_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _launch_environment(tmp_path, monkeypatch, device="cuda")

    assert environment[CUDA_TORCH_INTRAOP_THREADS_ENV] == "8"
    assert environment[CUDA_TORCH_INTEROP_THREADS_ENV] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a"


def test_cpu_worker_does_not_inherit_cuda_torch_thread_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _launch_environment(tmp_path, monkeypatch, device="cpu")

    assert CUDA_TORCH_INTRAOP_THREADS_ENV not in environment
    assert CUDA_TORCH_INTEROP_THREADS_ENV not in environment
