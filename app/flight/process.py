from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import time
import uuid
from collections.abc import Callable, Iterable, Mapping

from app.flight.records import (
    RecoverableAttemptRecord,
    recoverable_attempt_from_mapping,
)


_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
_PROC_ROOT = "/proc"


class ProcessRecoveryError(RuntimeError):
    """A recorded worker process group could not be recovered safely."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgrp: int
    session: int
    state: str
    boot_id: str
    start_ticks: int


@dataclass(frozen=True)
class ProcessRecoveryResult:
    job_id: str
    attempt: int
    pgid: int | None
    outcome: str
    escalated: bool = False


def read_boot_id(*, path: str = _BOOT_ID_PATH) -> str:
    try:
        with open(path, encoding="ascii") as source:
            value = source.read().strip().lower()
        parsed = uuid.UUID(value)
    except (OSError, ValueError) as exc:
        raise ProcessRecoveryError("Linux boot identity is unavailable") from exc
    if str(parsed) != value:
        raise ProcessRecoveryError("Linux boot identity is invalid")
    return value


def read_process_identity(
    pid: int,
    *,
    boot_id: str | None = None,
    proc_root: str = _PROC_ROOT,
) -> ProcessIdentity | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    try:
        with open(os.path.join(proc_root, str(pid), "stat"), "rb") as source:
            raw = source.read()
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None
    try:
        closing = raw.rindex(b")")
        parsed_pid = int(raw[: raw.index(b" ")])
        fields = raw[closing + 2 :].split()
        state = fields[0].decode("ascii")
        pgrp = int(fields[2])
        session = int(fields[3])
        start_ticks = int(fields[19])
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        raise ProcessRecoveryError(f"invalid /proc stat for pid {pid}") from exc
    if parsed_pid != pid or len(state) != 1 or start_ticks <= 0:
        raise ProcessRecoveryError(f"invalid process identity for pid {pid}")
    return ProcessIdentity(
        pid=pid,
        pgrp=pgrp,
        session=session,
        state=state,
        boot_id=boot_id or read_boot_id(),
        start_ticks=start_ticks,
    )


def capture_worker_process(pid: int) -> ProcessIdentity:
    """Capture and verify the identity created by ``start_new_session=True``."""
    boot_id = read_boot_id()
    identity = read_process_identity(pid, boot_id=boot_id)
    if identity is None:
        raise ProcessRecoveryError("worker subprocess exited before identity capture")
    if identity.pgrp != pid or identity.session != pid:
        raise ProcessRecoveryError("worker subprocess did not create an isolated session")
    return identity


def recover_process_groups(
    attempts: Iterable[RecoverableAttemptRecord | Mapping],
    *,
    grace_seconds: float,
    logger=None,
    signal_group: Callable[[int, int], None] = os.killpg,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    proc_root: str = _PROC_ROOT,
    boot_id_path: str = _BOOT_ID_PATH,
) -> tuple[ProcessRecoveryResult, ...]:
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")
    attempts = tuple(attempts)
    if not attempts:
        return ()
    current_boot_id = read_boot_id(path=boot_id_path)
    results = []
    for attempt in attempts:
        result = _recover_process_group(
            attempt,
            current_boot_id=current_boot_id,
            grace_seconds=float(grace_seconds),
            signal_group=signal_group,
            monotonic=monotonic,
            wait=wait,
            proc_root=proc_root,
        )
        results.append(result)
        if logger is not None:
            logger.event(
                "flight.recovery.process_group",
                jobId=result.job_id,
                attempt=result.attempt,
                pgid=result.pgid,
                outcome=result.outcome,
                escalated=result.escalated,
            )
    return tuple(results)


def _recover_process_group(
    attempt: RecoverableAttemptRecord | Mapping,
    *,
    current_boot_id: str,
    grace_seconds: float,
    signal_group,
    monotonic,
    wait,
    proc_root: str,
) -> ProcessRecoveryResult:
    if not isinstance(attempt, RecoverableAttemptRecord):
        attempt = recoverable_attempt_from_mapping(attempt)
    job_id = attempt.job_id
    attempt_number = attempt.attempt
    pid = attempt.pid
    pgid = attempt.pgid
    recorded_boot_id = attempt.boot_id
    recorded_start_ticks = attempt.process_start_ticks
    if pid is None and pgid is None:
        return ProcessRecoveryResult(job_id, attempt_number, None, "not_spawned")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(pgid, bool)
        or not isinstance(pgid, int)
        or pgid <= 0
        or pgid != pid
    ):
        raise ProcessRecoveryError(
            f"job {job_id} attempt {attempt_number} has an unsafe process-group record"
        )
    leader = read_process_identity(pid, boot_id=current_boot_id, proc_root=proc_root)
    if recorded_boot_id is None or recorded_start_ticks is None:
        if not _live_group_members(pgid, pid, proc_root, current_boot_id):
            return ProcessRecoveryResult(job_id, attempt_number, pgid, "not_found")
        raise ProcessRecoveryError(
            f"job {job_id} attempt {attempt_number} lacks safe process identity"
        )
    if recorded_boot_id != current_boot_id:
        return ProcessRecoveryResult(job_id, attempt_number, pgid, "previous_boot")
    if leader is not None and (
        leader.start_ticks != recorded_start_ticks
        or leader.pgrp != pgid
        or leader.session != pid
    ):
        # The numeric PID has been reused. Never signal a process that does not
        # exactly match the durable boot/start/session identity.
        return ProcessRecoveryResult(job_id, attempt_number, pgid, "identity_mismatch")

    members = _live_group_members(pgid, pid, proc_root, current_boot_id)
    if not members:
        return ProcessRecoveryResult(job_id, attempt_number, pgid, "not_found")
    _signal(signal_group, pgid, signal.SIGTERM)
    deadline = monotonic() + grace_seconds
    while monotonic() < deadline:
        if not _live_group_members(pgid, pid, proc_root, current_boot_id):
            return ProcessRecoveryResult(job_id, attempt_number, pgid, "terminated")
        wait(min(0.02, max(0.0, deadline - monotonic())))
    members = _live_group_members(pgid, pid, proc_root, current_boot_id)
    if not members:
        return ProcessRecoveryResult(job_id, attempt_number, pgid, "terminated")
    _signal(signal_group, pgid, signal.SIGKILL)
    kill_deadline = monotonic() + max(1.0, grace_seconds)
    while monotonic() < kill_deadline:
        if not _live_group_members(pgid, pid, proc_root, current_boot_id):
            return ProcessRecoveryResult(
                job_id,
                attempt_number,
                pgid,
                "killed",
                escalated=True,
            )
        wait(0.02)
    raise ProcessRecoveryError(
        f"job {job_id} attempt {attempt_number} process group {pgid} survived SIGKILL"
    )


def _live_group_members(
    pgid: int,
    session: int,
    proc_root: str,
    boot_id: str,
) -> tuple[ProcessIdentity, ...]:
    try:
        entries = os.listdir(proc_root)
    except OSError as exc:
        raise ProcessRecoveryError("Linux process table is unavailable") from exc
    members = []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            identity = read_process_identity(
                int(entry),
                boot_id=boot_id,
                proc_root=proc_root,
            )
        except OSError:
            # Processes can disappear or become unreadable while /proc is
            # scanned. An unreadable unrelated entry is not a recovery error.
            continue
        except ProcessRecoveryError:
            # A malformed process table entry makes it unsafe to conclude that
            # the recorded group is gone.
            raise
        if (
            identity is not None
            and identity.pgrp == pgid
            and identity.session == session
            and identity.state != "Z"
        ):
            members.append(identity)
    return tuple(members)


def _signal(signal_group, pgid: int, signum: int) -> None:
    try:
        signal_group(pgid, signum)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise ProcessRecoveryError(
            f"permission denied signalling orphan process group {pgid}"
        ) from exc


__all__ = [
    "ProcessIdentity",
    "ProcessRecoveryError",
    "ProcessRecoveryResult",
    "capture_worker_process",
    "read_boot_id",
    "read_process_identity",
    "recover_process_groups",
]
