import os
import select
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from app.flight.process import (
    capture_worker_process,
    read_process_identity,
    recover_process_groups,
)


def _record(identity):
    return {
        "job_id": str(uuid.uuid4()),
        "attempt": 1,
        "pid": identity.pid,
        "pgid": identity.pgrp,
        "boot_id": identity.boot_id,
        "process_start_ticks": identity.start_ticks,
    }


def _terminated(pid: int) -> bool:
    identity = read_process_identity(pid)
    return identity is None or identity.state == "Z"


def _wait_terminated(*pids: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(_terminated(pid) for pid in pids):
            return
        time.sleep(0.02)
    raise AssertionError(f"processes did not terminate: {pids}")


def _readline(stream, *, timeout: float = 5.0):
    readable, _, _ = select.select([stream], [], [], timeout)
    if not readable:
        raise AssertionError("subprocess did not produce a line before timeout")
    line = stream.readline()
    if not line:
        raise AssertionError("subprocess closed output before producing a line")
    return line


def test_recovery_kills_leader_dead_process_group_descendants():
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    leader_code = (
        "import subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}], "
        "stdout=subprocess.PIPE); "
        "child.stdout.readline(); "
        "print(child.pid, flush=True); "
        "sys.stdin.buffer.read(1)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        child_pid = int(_readline(leader.stdout))
        identity = capture_worker_process(leader.pid)
        leader.stdin.write(b"x")
        leader.stdin.flush()
        leader.stdin.close()
        leader.wait(timeout=5)
        assert read_process_identity(leader.pid) is None

        results = recover_process_groups([_record(identity)], grace_seconds=0.05)

        assert len(results) == 1
        assert results[0].outcome == "killed"
        assert results[0].escalated is True
        _wait_terminated(child_pid)
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if leader.poll() is None:
            leader.kill()
        leader.wait(timeout=5)


def test_recovery_never_signals_reused_pid_identity():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    identity = capture_worker_process(process.pid)
    record = {
        **_record(identity),
        "process_start_ticks": identity.start_ticks + 1,
    }
    signals = []
    try:
        result = recover_process_groups(
            [record],
            grace_seconds=0,
            signal_group=lambda pgid, signum: signals.append((pgid, signum)),
        )[0]

        assert result.outcome == "identity_mismatch"
        assert signals == []
        assert process.poll() is None
    finally:
        try:
            os.killpg(identity.pgrp, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_process_supervisor_kills_cli_group_when_worker_parent_dies(tmp_path):
    pid_file = tmp_path / "cli.pid"
    child_code = (
        "import os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent_code = r"""
import os
import pathlib
import signal
import subprocess
import sys
import time

pid_file = sys.argv[1]
project_root = sys.argv[2]
child_code = sys.argv[3]
supervisor_path = sys.argv[4]
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
supervisor = subprocess.Popen(
    [
        sys.executable,
        supervisor_path,
        str(os.getpid()),
        "--",
        sys.executable,
        "-c",
        child_code,
        pid_file,
    ],
    cwd=project_root,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
deadline = time.time() + 5
while time.time() < deadline and not pathlib.Path(pid_file).exists():
    if supervisor.poll() is not None:
        raise SystemExit(supervisor.returncode)
    time.sleep(0.01)
if not pathlib.Path(pid_file).exists():
    raise SystemExit(99)
print(supervisor.pid, flush=True)
"""
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(pid_file),
            str(Path(__file__).resolve().parents[1]),
            child_code,
            str(
                    Path(__file__).resolve().parents[1]
                    / "app"
                    / "service"
                    / "adapters"
                    / "outbound"
                    / "worker_process"
                    / "process_supervisor.py"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    supervisor_pid = None
    try:
        supervisor_pid = int(_readline(parent.stdout))
        cli_pid = int(pid_file.read_text())
        assert parent.wait(timeout=5) == 0
        _wait_terminated(supervisor_pid, cli_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if supervisor_pid is not None:
            try:
                os.killpg(supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
