from __future__ import annotations

import errno
import hashlib
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v3 import (
    CONTRACT_VERSION,
    MAX_EVENT_BYTES,
    WorkerContractError,
    encode_control_message,
    load_document,
    parse_event,
)
from app.project import PROJECT_ROOT
from app.service.adapters.outbound.worker.process import (
    ProcessRecoveryError,
    capture_worker_process,
)
from app.service.application.ports.jobs import JobRepository
from app.service.application.ports.observability import EventLogger
from app.service.application.ports.workers import (
    ExecutionInput,
    ExecutionPlan,
    ExecutionResult,
)
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import ExecutionJobRecord

_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_LOG_BYTES = 16 * 1024 * 1024
_STDERR_TAIL_BYTES = 128 * 1024
_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}


class WorkerSubprocessError(AttemptExecutionError):
    pass


WorkerSubprocessResult = ExecutionResult


@dataclass
class _WorkerEventState:
    expected_sequence: int = 1
    ready: bool = False
    terminal: str | None = None
    completed_artifact: JsonObject | None = None
    error: tuple[str, str] | None = None


class _RunnerConfig(Protocol):
    @property
    def subprocess_timeout_seconds(self) -> float: ...

    @property
    def cancel_grace_seconds(self) -> float: ...


class _RunnerLedger(JobRepository, Protocol):
    def set_attempt_process(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        pid: int,
        pgid: int,
        boot_id: str,
        process_start_ticks: int,
    ) -> None: ...

    def get_job(self, job_id: str) -> Mapping[str, object] | None: ...

    def mark_input_waiting(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        next_ordinal: int,
        input_revision: int,
    ) -> bool: ...


class _RunnerSpool(Protocol):
    def attempt_stdout_path(self, job_id: str, attempt: int) -> str: ...

    def attempt_stderr_path(self, job_id: str, attempt: int) -> str: ...

    def attempt_result_manifest_path(
        self,
        job_id: str,
        attempt: int,
    ) -> str: ...

    def ensure_parent(self, path: str) -> None: ...


PopenFactory = Callable[..., subprocess.Popen[bytes]]
RecoveryPublisher = Callable[[ExecutionJobRecord, JsonObject], None]
StreamingInputProvider = Callable[
    [ExecutionJobRecord, int],
    tuple[ExecutionInput, ...],
]


class WorkerSubprocessRunner:
    """Own one worker subprocess from Popen through complete process reap."""

    def __init__(
        self,
        config: _RunnerConfig,
        ledger: _RunnerLedger,
        spool: _RunnerSpool,
        *,
        logger: EventLogger,
        popen_factory: PopenFactory = subprocess.Popen,
        signal_group: Callable[[int, int], None] = os.killpg,
        python_executable: str,
        publish_recovery: RecoveryPublisher | None = None,
        stream_inputs: StreamingInputProvider | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.logger = logger
        self._popen = popen_factory
        self._signal_group = signal_group
        self._python = python_executable
        self._publish_recovery = publish_recovery
        self._stream_inputs = stream_inputs
        self._monotonic = monotonic
        self._control_lock = threading.Lock()
        self._control_wakes: dict[str, threading.Event] = {}

    def notify_input(self, job_id: str) -> None:
        with self._control_lock:
            wake = self._control_wakes.get(job_id)
        if wake is not None:
            wake.set()

    def run(
        self,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        *,
        cancel: threading.Event,
        force_stop: threading.Event,
    ) -> WorkerSubprocessResult:
        if plan.protocol_version == CONTRACT_VERSION:
            return self._run_attempt(
                job,
                plan,
                cancel=cancel,
                force_stop=force_stop,
            )
        raise WorkerSubprocessError(
            ErrorCode.WORKER_PROTOCOL_VIOLATION,
            "unsupported internal worker protocol version",
        )

    def _run_attempt(
        self,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        *,
        cancel: threading.Event,
        force_stop: threading.Event,
    ) -> WorkerSubprocessResult:
        job_id = job.job_id
        attempt = job.attempt
        attempt_id = _active_attempt_id(job)
        stdout_path = self.spool.attempt_stdout_path(job_id, attempt)
        stderr_path = self.spool.attempt_stderr_path(job_id, attempt)
        self.spool.ensure_parent(stdout_path)
        self.spool.ensure_parent(stderr_path)
        Path(stdout_path).touch()

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if job.selected_device == "cuda":
            if not job.assigned_device_id:
                raise WorkerSubprocessError(
                    ErrorCode.INTERNAL,
                    "CUDA attempt has no assigned physical device",
                )
            environment["CUDA_VISIBLE_DEVICES"] = job.assigned_device_id
        supervised_argv = [
            self._python,
            os.path.join(
                PROJECT_ROOT,
                "app",
                "service",
                "adapters",
                "outbound",
                "worker",
                "supervisor.py",
            ),
            str(os.getpid()),
            "--",
            *plan.argv,
        ]
        try:
            process = self._popen(
                supervised_argv,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                close_fds=True,
                bufsize=0,
            )
        except OSError as exc:
            raise WorkerSubprocessError(
                ErrorCode.SUBPROCESS_FAILED,
                "worker subprocess could not be started",
            ) from exc

        errors: queue.Queue[WorkerSubprocessError] = queue.Queue()
        stderr_tail: list[bytes] = []
        event_state = _WorkerEventState()
        finished = threading.Event()
        control_wake = threading.Event()
        threads: list[threading.Thread] = []
        started_threads: list[threading.Thread] = []
        deadline = self._monotonic() + self.config.subprocess_timeout_seconds
        with self._control_lock:
            self._control_wakes[job_id] = control_wake
        try:
            try:
                identity = capture_worker_process(process.pid)
            except ProcessRecoveryError as exc:
                raise WorkerSubprocessError(
                    ErrorCode.INTERNAL,
                    "worker subprocess identity could not be captured safely",
                ) from exc
            self.ledger.set_attempt_process(
                job_id,
                attempt,
                attempt_id=attempt_id,
                pid=process.pid,
                pgid=process.pid,
                boot_id=identity.boot_id,
                process_start_ticks=identity.start_ticks,
            )
            self.logger.event(
                "flight.worker.started",
                jobId=job_id,
                attempt=attempt,
                attemptId=attempt_id,
                device=job.selected_device,
                workerPid=process.pid,
                inputs=len(plan.inputs),
                workerProtocol=plan.protocol_version,
            )
            threads = [
                threading.Thread(
                    target=self._read_worker_events,
                    args=(process.stdout, job, plan, event_state, errors),
                    name=f"worker-events-{job_id}",
                    daemon=True,
                ),
                threading.Thread(
                    target=self._drain_log,
                    args=(process.stderr, stderr_path, errors),
                    kwargs={"tail": stderr_tail},
                    name=f"worker-stderr-{job_id}",
                    daemon=True,
                ),
                threading.Thread(
                    target=self._feed_worker_controls,
                    args=(
                        process.stdin,
                        job,
                        plan,
                        control_wake,
                        finished,
                        errors,
                    ),
                    name=f"worker-controls-{job_id}",
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
                started_threads.append(thread)
        except BaseException:
            finished.set()
            control_wake.set()
            self._abort_spawned_process(process, started_threads)
            with self._control_lock:
                self._control_wakes.pop(job_id, None)
            raise

        failure: WorkerSubprocessError | None = None
        term_sent_at: float | None = None
        killed = False
        try:
            while process.poll() is None:
                now = self._monotonic()
                if failure is None and cancel.is_set():
                    failure = _cancellation_failure(force_stop)
                if failure is None:
                    try:
                        failure = errors.get_nowait()
                    except queue.Empty:
                        pass
                if failure is None and now >= deadline:
                    failure = WorkerSubprocessError(
                        ErrorCode.SUBPROCESS_HUNG,
                        "worker subprocess exceeded its execution timeout",
                    )
                if failure is not None and term_sent_at is None:
                    self._signal_process_group(process, signal.SIGTERM)
                    term_sent_at = now
                elif (
                    term_sent_at is not None
                    and not killed
                    and now - term_sent_at >= self.config.cancel_grace_seconds
                ):
                    self._signal_process_group(process, signal.SIGKILL)
                    killed = True
                time.sleep(0.02)
            exit_code = process.wait()
        finally:
            finished.set()
            control_wake.set()
            while any(thread.is_alive() for thread in started_threads):
                now = self._monotonic()
                if failure is None and cancel.is_set():
                    failure = _cancellation_failure(force_stop)
                if failure is None:
                    try:
                        failure = errors.get_nowait()
                    except queue.Empty:
                        pass
                if failure is None and now >= deadline:
                    failure = WorkerSubprocessError(
                        ErrorCode.SUBPROCESS_HUNG,
                        "worker subprocess pipes exceeded the execution timeout",
                    )
                if failure is not None and term_sent_at is None:
                    self._signal_process_group(process, signal.SIGTERM)
                    term_sent_at = now
                elif (
                    term_sent_at is not None
                    and not killed
                    and now - term_sent_at >= self.config.cancel_grace_seconds
                ):
                    self._signal_process_group(process, signal.SIGKILL)
                    killed = True
                for thread in started_threads:
                    thread.join(0.02)
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except OSError:
                    pass
            with self._control_lock:
                if self._control_wakes.get(job_id) is control_wake:
                    self._control_wakes.pop(job_id, None)

        while failure is None:
            try:
                failure = errors.get_nowait()
            except queue.Empty:
                break
        tail = stderr_tail[-1] if stderr_tail else b""
        classified_exit = (
            self._classify_subprocess_exit(exit_code, tail)
            if exit_code != 0
            else None
        )
        if classified_exit is not None and classified_exit.code in (
            ErrorCode.CUDA_OUT_OF_MEMORY,
            ErrorCode.DEVICE_UNAVAILABLE,
            ErrorCode.DEVICE_LOST,
        ):
            raise classified_exit
        if failure is not None:
            raise WorkerSubprocessError(
                failure.code,
                failure.message,
                exit_code,
            )
        if event_state.terminal == "completed":
            if exit_code != 0:
                raise WorkerSubprocessError(
                    ErrorCode.WORKER_PROTOCOL_VIOLATION,
                    "worker emitted completed but exited unsuccessfully",
                    exit_code,
                )
            result = self._load_result_manifest(
                job,
                plan,
                event_state.completed_artifact,
            )
            return WorkerSubprocessResult(
                exit_code,
                tail,
                result_manifest=result,
            )
        if event_state.terminal == "error":
            if exit_code == 0:
                raise WorkerSubprocessError(
                    ErrorCode.WORKER_PROTOCOL_VIOLATION,
                    "worker emitted error but exited successfully",
                    exit_code,
                )
            code_text, message = event_state.error or (
                ErrorCode.SUBPROCESS_FAILED.value,
                "worker execution failed",
            )
            try:
                code = ErrorCode(code_text)
            except ValueError:
                code = ErrorCode.SUBPROCESS_FAILED
            raise WorkerSubprocessError(code, message, exit_code)
        if classified_exit is not None:
            raise classified_exit
        raise WorkerSubprocessError(
            ErrorCode.WORKER_PROTOCOL_VIOLATION,
            "worker exited without a terminal event",
            exit_code,
        )

    def _feed_worker_controls(
        self,
        stream: BinaryIO,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        wake: threading.Event,
        finished: threading.Event,
        errors: queue.Queue[WorkerSubprocessError],
    ) -> None:
        attempt_id = _active_attempt_id(job)
        sequence = 0
        next_ordinal = len(plan.inputs)
        try:
            if self._stream_inputs is None:
                raise WorkerSubprocessError(
                    ErrorCode.INTERNAL,
                    "streaming input provider is unavailable",
                )
            if job.input_state == InputState.CLOSED:
                stream.close()
                return
            while not finished.is_set():
                current = self.ledger.get_execution_job(job.job_id)
                if (
                    current is None
                    or current.attempt != job.attempt
                    or current.attempt_id != attempt_id
                    or current.execution_state
                    not in (
                        ExecutionState.RUNNING,
                        ExecutionState.CANCELLING,
                    )
                ):
                    raise WorkerSubprocessError(
                        ErrorCode.EXECUTION_INTERRUPTED,
                        "worker attempt no longer owns the input stream",
                    )
                for item in self._stream_inputs(current, next_ordinal):
                    sequence += 1
                    stream.write(encode_control_message(
                        job_id=job.job_id,
                        attempt=job.attempt,
                        attempt_id=attempt_id,
                        sequence=sequence,
                        message_type="input.committed",
                        payload={
                            "inputRevision": current.input_revision,
                            "input": _worker_input_manifest(item),
                        },
                    ))
                    stream.flush()
                    next_ordinal = item.ordinal + 1
                if current.input_state == InputState.CLOSED:
                    row = self.ledger.get_job(job.job_id)
                    if row is None:
                        raise WorkerSubprocessError(
                            ErrorCode.INTERNAL,
                            "closed input summary is unavailable",
                        )
                    manifest_sha256 = _optional_string(
                        row.get("manifest_sha256"),
                        "closed input manifestSha256",
                    )
                    if manifest_sha256 is None:
                        raise WorkerSubprocessError(
                            ErrorCode.INTERNAL,
                            "closed input summary is unavailable",
                        )
                    sequence += 1
                    stream.write(encode_control_message(
                        job_id=job.job_id,
                        attempt=job.attempt,
                        attempt_id=attempt_id,
                        sequence=sequence,
                        message_type="input.closed",
                        payload={
                            "inputRevision": _integer(
                                row.get("input_revision"),
                                "closed input revision",
                            ),
                            "payloadCount": _integer(
                                row.get("payload_count"),
                                "closed input payload count",
                            ),
                            "totalRows": _integer(
                                row.get("total_rows"),
                                "closed input row count",
                            ),
                            "totalBytes": _integer(
                                row.get("total_bytes"),
                                "closed input byte count",
                            ),
                            "manifestSha256": manifest_sha256,
                        },
                    ))
                    stream.flush()
                    stream.close()
                    return
                wake.wait(0.25)
                wake.clear()
        except BrokenPipeError:
            if not finished.is_set():
                errors.put(WorkerSubprocessError(
                    ErrorCode.WORKER_PROTOCOL_VIOLATION,
                    "worker closed its control channel before input EOF",
                ))
        except AttemptExecutionError as exc:
            errors.put(WorkerSubprocessError(exc.code, exc.message))
        except (OSError, ValueError, WorkerContractError) as exc:
            errors.put(WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                str(exc) or "worker control stream failed",
            ))
        except Exception:
            errors.put(WorkerSubprocessError(
                ErrorCode.INTERNAL,
                "worker control processing failed internally",
            ))

    def _read_worker_events(
        self,
        stream: BinaryIO,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        state: _WorkerEventState,
        errors: queue.Queue[WorkerSubprocessError],
    ) -> None:
        attempt_id = _active_attempt_id(job)
        try:
            while True:
                line = stream.readline(MAX_EVENT_BYTES + 2)
                if line == b"":
                    return
                event = parse_event(line)
                if (
                    _string(event["jobId"], "event jobId") != job.job_id
                    or _integer(event["attempt"], "event attempt")
                    != job.attempt
                    or _string(event["attemptId"], "event attemptId")
                    != attempt_id
                ):
                    raise WorkerContractError(
                        "worker event identity does not match the active attempt"
                    )
                if (
                    _integer(event["sequence"], "event sequence")
                    != state.expected_sequence
                ):
                    raise WorkerContractError(
                        "worker event sequence is not contiguous"
                    )
                state.expected_sequence += 1
                event_type = _string(event["type"], "event type")
                payload = _object(event["payload"], "event payload")
                if state.terminal is not None:
                    raise WorkerContractError(
                        "worker emitted an event after its terminal event"
                    )
                if not state.ready:
                    if event_type != "ready":
                        raise WorkerContractError(
                            "worker ready must be the first event"
                        )
                    if (
                        _integer(
                            payload.get("nextOrdinal"),
                            "ready nextOrdinal",
                        )
                        != len(plan.inputs)
                        or _integer(
                            payload.get("inputRevision"),
                            "ready inputRevision",
                        )
                        != job.input_revision
                    ):
                        raise WorkerContractError(
                            "worker ready snapshot differs from the command manifest"
                        )
                    state.ready = True
                    continue
                if event_type == "ready":
                    raise WorkerContractError("worker emitted ready more than once")
                if event_type == "input.ack":
                    continue
                if event_type == "input.waiting":
                    registered = self.ledger.mark_input_waiting(
                        job.job_id,
                        job.attempt,
                        attempt_id=attempt_id,
                        next_ordinal=_integer(
                            payload.get("nextOrdinal"),
                            "waiting nextOrdinal",
                        ),
                        input_revision=_integer(
                            payload.get("inputRevision"),
                            "waiting inputRevision",
                        ),
                    )
                    if not registered:
                        self.notify_input(job.job_id)
                    continue
                if event_type == "progress":
                    try:
                        self.ledger.update_progress(
                            job.job_id,
                            _object(
                                payload.get("progress"),
                                "progress payload",
                            ),
                            attempt_id=attempt_id,
                        )
                    except ServiceError as exc:
                        if self._progress_rejected_by_cancel(job):
                            # A committed cancel makes progress immutable. Keep
                            # draining until the cancellation signal stops the
                            # process.
                            continue
                        raise WorkerSubprocessError(
                            ErrorCode.EXECUTION_INTERRUPTED,
                            "worker attempt no longer owns job progress",
                        ) from exc
                elif event_type == "checkpoint":
                    if self._publish_recovery is None:
                        raise WorkerContractError(
                            "worker emitted an unexpected checkpoint"
                        )
                    self._publish_recovery(job, payload)
                elif event_type == "completed":
                    state.terminal = event_type
                    state.completed_artifact = _object(
                        payload.get("resultManifest"),
                        "completed resultManifest",
                    )
                elif event_type == "error":
                    state.terminal = event_type
                    state.error = (
                        _string(payload.get("code"), "worker error code"),
                        _string(
                            payload.get("message"),
                            "worker error message",
                        ),
                    )
        except AttemptExecutionError as exc:
            errors.put(WorkerSubprocessError(exc.code, exc.message))
        except (OSError, WorkerContractError, ValueError) as exc:
            errors.put(WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                str(exc) or "worker event stream is invalid",
            ))
        except Exception:
            errors.put(WorkerSubprocessError(
                ErrorCode.INTERNAL,
                "worker event processing failed internally",
            ))

    def _load_result_manifest(
        self,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        artifact: JsonObject | None,
    ) -> JsonObject:
        if artifact is None:
            raise WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                "worker completed event has no result manifest",
            )
        expected = self.spool.attempt_result_manifest_path(
            job.job_id,
            job.attempt,
        )
        path = os.path.abspath(
            _string(artifact.get("path"), "result manifest path")
        )
        if path != expected:
            raise WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                "worker result manifest path is not managed by the attempt",
            )
        try:
            if (
                os.path.getsize(path)
                != _integer(
                    artifact.get("byteCount"),
                    "result manifest byte count",
                )
                or _sha256_file(path)
                != _string(
                    artifact.get("sha256"),
                    "result manifest sha256",
                )
            ):
                raise ValueError("result manifest integrity check failed")
            result = load_document(path, "result-manifest")
        except (OSError, ValueError, WorkerContractError) as exc:
            raise WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                "worker result manifest is invalid",
            ) from exc
        if (
            _string(result["jobId"], "result jobId") != job.job_id
            or _integer(result["attempt"], "result attempt") != job.attempt
            or _string(result["attemptId"], "result attemptId")
            != _active_attempt_id(job)
            or _string(result["operation"], "result operation")
            != job.operation
        ):
            raise WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                "worker result identity does not match the active attempt",
            )
        current = self.ledger.get_job(job.job_id)
        if (
            current is None
            or _string(
                current.get("input_state") if current else None,
                "current input state",
            )
            != InputState.CLOSED.value
            or _integer(result["inputRevision"], "result inputRevision")
            != _integer(
                current.get("input_revision") if current else None,
                "current input revision",
            )
            or _string(result["manifestSha256"], "result manifestSha256")
            != _string(
                current.get("manifest_sha256") if current else None,
                "current manifest sha256",
            )
        ):
            raise WorkerSubprocessError(
                ErrorCode.WORKER_PROTOCOL_VIOLATION,
                "worker result input identity differs from closed input",
            )
        return result

    def _drain_log(
        self,
        stream: BinaryIO,
        path: str,
        errors: queue.Queue[WorkerSubprocessError],
        *,
        tail: list[bytes] | None = None,
    ) -> None:
        persisted = 0
        captured = bytearray()
        target: BinaryIO | None = None
        try:
            target = open(path, "wb")
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                if tail is not None:
                    captured.extend(chunk)
                    if len(captured) > _STDERR_TAIL_BYTES:
                        del captured[:-_STDERR_TAIL_BYTES]
                if persisted < _MAX_LOG_BYTES:
                    kept = chunk[: _MAX_LOG_BYTES - persisted]
                    target.write(kept)
                    target.flush()
                    persisted += len(kept)
        except OSError as exc:
            code = (
                ErrorCode.DISK_FULL
                if exc.errno in _DISK_FULL_ERRNOS
                else ErrorCode.INTERNAL
            )
            errors.put(WorkerSubprocessError(
                code,
                "worker log could not be persisted",
            ))
            # Continue draining even after a logging failure to avoid deadlock.
            try:
                while stream.read(64 * 1024):
                    pass
            except OSError:
                pass
        finally:
            if target is not None:
                target.close()
            if tail is not None:
                tail.append(bytes(captured))

    def _progress_rejected_by_cancel(
        self,
        job: ExecutionJobRecord,
    ) -> bool:
        current = self.ledger.get_execution_job(job.job_id)
        return (
            current is not None
            and current.attempt_id == _active_attempt_id(job)
            and current.execution_state
            in (ExecutionState.CANCELLING, ExecutionState.CANCELLED)
        )

    def _signal_process_group(
        self,
        process: subprocess.Popen[bytes],
        signum: int,
    ) -> None:
        try:
            self._signal_group(process.pid, signum)
        except ProcessLookupError:
            pass

    def _abort_spawned_process(
        self,
        process: subprocess.Popen[bytes],
        threads: Sequence[threading.Thread],
    ) -> None:
        """Kill and reap a child if worker setup fails after Popen."""
        self._signal_process_group(process, signal.SIGKILL)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass
        try:
            process.wait(
                timeout=max(1.0, self.config.cancel_grace_seconds)
            )
        except subprocess.TimeoutExpired:
            self._signal_process_group(process, signal.SIGKILL)
            process.wait()
        for thread in threads:
            thread.join()

    @staticmethod
    def _classify_subprocess_exit(
        exit_code: int,
        stderr_tail: bytes,
    ) -> WorkerSubprocessError:
        text = stderr_tail.lower()
        if b"training recovery" in text:
            return WorkerSubprocessError(
                ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
                "training recovery checkpoint could not be restored",
                exit_code,
            )
        if b"cuda" in text and (
            b"out of memory" in text or b"outofmemoryerror" in text
        ):
            return WorkerSubprocessError(
                ErrorCode.CUDA_OUT_OF_MEMORY,
                "CUDA execution ran out of memory",
                exit_code,
            )
        if b"cuda" in text and any(fragment in text for fragment in (
            b"not available",
            b"driver shutting down",
            b"no cuda gpus",
            b"device has been lost",
            b"cuda_error_device_lost",
            b"cuda error: unknown error",
            b"cuda_error_unknown",
            b"cuda error: initialization error",
            b"cudaerrorinitializationerror",
            b"cuda driver error",
        )):
            return WorkerSubprocessError(
                ErrorCode.DEVICE_LOST,
                "CUDA device became unavailable during execution",
                exit_code,
            )
        return WorkerSubprocessError(
            ErrorCode.SUBPROCESS_FAILED,
            f"worker subprocess exited with status {exit_code}",
            exit_code,
        )


def _cancellation_failure(
    force_stop: threading.Event,
) -> WorkerSubprocessError:
    code = (
        ErrorCode.EXECUTION_INTERRUPTED
        if force_stop.is_set()
        else ErrorCode.CANCELLED
    )
    message = (
        "worker execution was interrupted by service shutdown"
        if code == ErrorCode.EXECUTION_INTERRUPTED
        else "job cancellation was requested"
    )
    return WorkerSubprocessError(code, message)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_input_manifest(item: ExecutionInput) -> JsonObject:
    return {
        "schemaId": item.schema_id,
        "ordinal": item.ordinal,
        "commitRevision": item.commit_revision,
        "dataContractSha256": item.data_contract_sha256,
        "rows": item.rows,
        "artifact": {
            "path": item.absolute_path,
            "byteCount": item.byte_count,
            "sha256": item.sha256,
        },
    }


def _active_attempt_id(job: ExecutionJobRecord) -> str:
    if job.attempt_id is None:
        raise WorkerSubprocessError(
            ErrorCode.INTERNAL,
            "claimed job has no attempt identity",
        )
    return job.attempt_id


def _object(value: JsonValue, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise WorkerContractError(f"{label} must be an object")
    return cast(JsonObject, value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise WorkerContractError(f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise WorkerContractError(f"{label} must be a string or null")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerContractError(f"{label} must be an integer")
    return value
