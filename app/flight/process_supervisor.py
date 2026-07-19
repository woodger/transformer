"""Linux parent-death guard for a worker-owned legacy CLI subprocess."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys


_PR_SET_PDEATHSIG = 1
_PARENT_DEATH_SIGNAL = signal.SIGUSR1


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 3 or arguments[1] != "--":
        return 64
    try:
        expected_parent = int(arguments[0])
    except ValueError:
        return 64
    command = arguments[2:]
    if expected_parent <= 0 or not command or any("\x00" in item for item in command):
        return 64
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        # Never risk signalling the caller's process group when this internal
        # module is invoked outside WorkerPool's start_new_session boundary.
        return 70

    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {_PARENT_DEATH_SIGNAL})
    except (AttributeError, OSError):
        return 70
    signal.signal(_PARENT_DEATH_SIGNAL, _kill_own_group)
    if os.getppid() != expected_parent:
        _kill_own_group()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, _PARENT_DEATH_SIGNAL, 0, 0, 0) != 0:
            return 70
    except (AttributeError, OSError):
        return 70
    # The parent may exit between the first getppid() and prctl(). Linux does
    # not retroactively deliver PDEATHSIG, so this second check is required.
    if os.getppid() != expected_parent:
        _kill_own_group()

    try:
        child = subprocess.Popen(
            command,
            shell=False,
            start_new_session=False,
            close_fds=True,
        )
    except OSError:
        return 127
    return_code = child.wait()
    return return_code if return_code >= 0 else 128 - return_code


def _kill_own_group(_signum=None, _frame=None) -> None:
    os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(128 + signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
