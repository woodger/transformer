from __future__ import annotations

import errno
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT
from app.flight.constants import ErrorCode
from app.flight.errors import ServiceError
from app.flight.process import ProcessRecoveryError, capture_worker_process
from app.flight.records import ExecutionJobRecord
from app.flight.worker_artifacts import (
    StagedPredictionOutput,
    WorkerArtifactError,
)
from app.flight.worker_plan import ExecutionInput, ExecutionPlan
from app.flight.worker_recovery import WorkerRecoveryError

_FRAME_HEADER_BYTES = 8
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_LOG_BYTES = 16 * 1024 * 1024
_STDERR_TAIL_BYTES = 128 * 1024
_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}


@dataclass(frozen=True)
class WorkerSubprocessError(Exception):
    code: ErrorCode
    message: str
    exit_code: int | None = None

    def __post_init__(self):
        Exception.__init__(self, self.message)


@dataclass(frozen=True)
class WorkerSubprocessResult:
    exit_code: int
    stderr_tail: bytes
    outputs: tuple[StagedPredictionOutput, ...] = ()


class WorkerSubprocessRunner:
    """Own one worker subprocess from Popen through complete process reap."""

    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        logger,
        popen_factory: Callable = subprocess.Popen,
        signal_group: Callable[[int, int], None] = os.killpg,
        python_executable: str,
        stage_prediction: Callable,
        publish_recovery: Callable | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.logger = logger
        self._popen = popen_factory
        self._signal_group = signal_group
        self._python = python_executable
        self._stage_prediction = stage_prediction
        self._publish_recovery = publish_recovery
        self._monotonic = monotonic

    def run(
        self,
        job: ExecutionJobRecord,
        plan: ExecutionPlan,
        *,
        cancel: threading.Event,
        force_stop: threading.Event,
    ) -> WorkerSubprocessResult:
        job_id = job.job_id
        attempt = job.attempt
        inputs = plan.inputs
        argv = plan.argv
        stdout_path = self.spool.attempt_stdout_path(job_id, attempt)
        stderr_path = self.spool.attempt_stderr_path(job_id, attempt)
        self.spool.ensure_parent(stdout_path)
        self.spool.ensure_parent(stderr_path)

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
                "flight",
                "process_supervisor.py",
            ),
            str(os.getpid()),
            "--",
            *argv,
        ]
        try:
            process = self._popen(
                supervised_argv,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=(
                    subprocess.DEVNULL
                    if plan.uses_spooled_fit
                    else subprocess.PIPE
                ),
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
        deadline = self._monotonic() + self.config.subprocess_timeout_seconds
        errors: queue.Queue[WorkerSubprocessError] = queue.Queue()
        finished = threading.Event()
        stderr_tail: list[bytes] = []
        outputs: list[StagedPredictionOutput] = []
        threads: list[threading.Thread] = []
        started_threads: list[threading.Thread] = []
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
                pid=process.pid,
                pgid=process.pid,
                boot_id=identity.boot_id,
                process_start_ticks=identity.start_ticks,
            )
            self.logger.event(
                "flight.worker.started",
                jobId=job_id,
                attempt=attempt,
                device=job.selected_device,
                workerPid=process.pid,
                inputs=len(inputs),
            )
            threads = [
                threading.Thread(
                    target=self._drain_log,
                    args=(process.stderr, stderr_path, errors),
                    kwargs={"tail": stderr_tail},
                    name=f"flight-stderr-{job_id}",
                    daemon=True,
                ),
            ]
            if not plan.uses_spooled_fit:
                threads.insert(
                    0,
                    threading.Thread(
                        target=self._feed_frames,
                        args=(process.stdin, inputs, errors),
                        name=f"flight-stdin-{job_id}",
                        daemon=True,
                    ),
                )
            if job.operation == "predict":
                # Keep a deterministic empty text log; binary stdout is staged
                # as Arrow.
                Path(stdout_path).touch()
                threads.append(threading.Thread(
                    target=self._read_prediction_frames,
                    args=(process.stdout, job, inputs, outputs, errors),
                    name=f"flight-stdout-{job_id}",
                    daemon=True,
                ))
            else:
                threads.append(threading.Thread(
                    target=self._drain_log,
                    args=(process.stdout, stdout_path, errors),
                    name=f"flight-stdout-{job_id}",
                    daemon=True,
                ))
                if (
                    plan.uses_training_recovery
                    and self._publish_recovery is not None
                ):
                    threads.append(threading.Thread(
                        target=self._tail_recovery_events,
                        args=(job, finished, errors),
                        name=f"flight-recovery-{job_id}",
                        daemon=True,
                    ))
                threads.append(threading.Thread(
                    target=self._tail_metrics,
                    args=(
                        job,
                        inputs,
                        finished,
                        errors,
                        plan.resume_training_complete,
                    ),
                    name=f"flight-metrics-{job_id}",
                    daemon=True,
                ))
            for thread in threads:
                thread.start()
                started_threads.append(thread)
        except BaseException:
            finished.set()
            self._abort_spawned_process(process, started_threads)
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
                    and now - term_sent_at
                    >= self.config.cancel_grace_seconds
                ):
                    self._signal_process_group(process, signal.SIGKILL)
                    killed = True
                time.sleep(0.02)
            exit_code = process.wait()
        finally:
            finished.set()
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except OSError:
                pass
            # A descendant can inherit stdout/stderr and keep the pipe open
            # after the direct child exits. Continue enforcing the same
            # execution deadline while pumps drain, and signal the whole
            # process group even when its original leader has already exited.
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
                    and now - term_sent_at
                    >= self.config.cancel_grace_seconds
                ):
                    self._signal_process_group(process, signal.SIGKILL)
                    killed = True
                for thread in started_threads:
                    thread.join(0.02)
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except OSError:
                    pass

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
        if classified_exit is not None:
            raise classified_exit
        return WorkerSubprocessResult(exit_code, tail, tuple(outputs))

    def _feed_frames(
        self,
        stream,
        inputs: tuple[ExecutionInput, ...],
        errors,
    ) -> None:
        try:
            for item in inputs:
                size = item.byte_count
                stream.write(
                    size.to_bytes(_FRAME_HEADER_BYTES, "big", signed=False)
                )
                with open(item.absolute_path, "rb") as source:
                    shutil.copyfileobj(source, stream, _COPY_CHUNK_BYTES)
                stream.flush()
            stream.write(
                (0).to_bytes(_FRAME_HEADER_BYTES, "big", signed=False)
            )
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            errors.put(WorkerSubprocessError(
                ErrorCode.SUBPROCESS_FAILED,
                "worker subprocess input stream failed",
            ))
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _drain_log(self, stream, path: str, errors, *, tail=None) -> None:
        persisted = 0
        captured = bytearray()
        target = None
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

    def _read_prediction_frames(
        self,
        stream,
        job: ExecutionJobRecord,
        inputs: tuple[ExecutionInput, ...],
        outputs: list[StagedPredictionOutput],
        errors,
    ) -> None:
        try:
            for item in inputs:
                header = _read_exact(stream, _FRAME_HEADER_BYTES)
                size = int.from_bytes(header, "big", signed=False)
                if size <= 0:
                    raise WorkerSubprocessError(
                        ErrorCode.MALFORMED_OUTPUT,
                        "prediction subprocess emitted a zero-length frame",
                    )
                if size > self.config.max_payload_bytes:
                    raise WorkerSubprocessError(
                        ErrorCode.MALFORMED_OUTPUT,
                        "prediction subprocess output exceeded the payload limit",
                    )
                output = self._stage_prediction(
                    stream,
                    job,
                    item,
                    size,
                )
                outputs.append(output)
            if stream.read(1) != b"":
                raise WorkerSubprocessError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction subprocess emitted more frames than expected",
                )
        except WorkerSubprocessError as exc:
            errors.put(exc)
        except WorkerArtifactError as exc:
            errors.put(WorkerSubprocessError(exc.code, exc.message))
        except OSError as exc:
            if exc.errno in _DISK_FULL_ERRNOS:
                errors.put(WorkerSubprocessError(
                    ErrorCode.DISK_FULL,
                    "prediction output could not be persisted",
                ))
            else:
                errors.put(WorkerSubprocessError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction subprocess emitted malformed Arrow output",
                ))
        except (EOFError, ServiceError, ValueError):
            errors.put(WorkerSubprocessError(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess emitted malformed Arrow output",
            ))

    def _tail_metrics(
        self,
        job: ExecutionJobRecord,
        inputs: tuple[ExecutionInput, ...],
        finished,
        errors,
        allow_empty: bool = False,
    ) -> None:
        path = self.spool.attempt_metrics_path(job.job_id, job.attempt)
        offset = 0
        inode = None
        partial = b""
        records = 0
        try:
            while True:
                # A process exit makes its metrics file stable, but the waiter
                # may set ``finished`` while this iteration is between stat()
                # and read(). Only finalize when the process was already
                # finished before the iteration began; otherwise perform one
                # final pass over the now-stable file.
                finished_at_iteration_start = finished.is_set()
                try:
                    stat = os.stat(path)
                except FileNotFoundError:
                    if finished_at_iteration_start:
                        if allow_empty:
                            return
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess did not produce metrics JSONL",
                        ) from None
                    finished.wait(0.05)
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity != inode or stat.st_size < offset:
                    inode = identity
                    offset = 0
                    partial = b""
                with open(path, "rb") as source:
                    source.seek(offset)
                    chunk = source.read()
                    offset = source.tell()
                partial += chunk
                lines = partial.split(b"\n")
                partial = lines.pop()
                for line in lines:
                    if not line:
                        continue
                    try:
                        progress = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess emitted malformed metrics JSONL",
                        ) from exc
                    if not _valid_progress(progress):
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess emitted malformed metrics JSONL",
                        )
                    frame = progress.get("frame")
                    if isinstance(frame, int) and 1 <= frame <= len(inputs):
                        progress["ordinal"] = inputs[frame - 1].ordinal
                    self.ledger.update_progress(job.job_id, progress)
                    records += 1
                if finished_at_iteration_start:
                    if partial.strip():
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess left an incomplete metrics record",
                        )
                    if records == 0 and not allow_empty:
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess produced no metrics records",
                        )
                    return
                finished.wait(0.05)
        except WorkerSubprocessError as exc:
            errors.put(exc)
        except ServiceError:
            # Cancellation/publication may make progress immutable.
            return
        except OSError as exc:
            code = (
                ErrorCode.DISK_FULL
                if exc.errno in _DISK_FULL_ERRNOS
                else ErrorCode.INTERNAL
            )
            errors.put(WorkerSubprocessError(
                code,
                "fit progress could not be read",
            ))
        except Exception:
            errors.put(WorkerSubprocessError(
                ErrorCode.INTERNAL,
                "fit progress processing failed internally",
            ))

    def _tail_recovery_events(
        self,
        job: ExecutionJobRecord,
        finished,
        errors,
    ) -> None:
        path = self.spool.attempt_recovery_events_path(
            job.job_id,
            job.attempt,
        )
        offset = 0
        partial = b""
        try:
            while True:
                finished_at_iteration_start = finished.is_set()
                try:
                    with open(path, "rb") as source:
                        source.seek(offset)
                        chunk = source.read()
                        offset = source.tell()
                except FileNotFoundError:
                    if finished_at_iteration_start:
                        return
                    finished.wait(0.05)
                    continue
                partial += chunk
                lines = partial.split(b"\n")
                partial = lines.pop()
                for line in lines:
                    if not line:
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess emitted malformed recovery events",
                        ) from exc
                    try:
                        self._publish_recovery(job, event)
                    except WorkerRecoveryError as exc:
                        raise WorkerSubprocessError(
                            exc.code,
                            exc.message,
                        ) from exc
                if finished_at_iteration_start:
                    if partial.strip():
                        raise WorkerSubprocessError(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess left an incomplete recovery event",
                        )
                    return
                finished.wait(0.05)
        except WorkerSubprocessError as exc:
            errors.put(exc)
        except OSError as exc:
            code = (
                ErrorCode.DISK_FULL
                if exc.errno in _DISK_FULL_ERRNOS
                else ErrorCode.INTERNAL
            )
            errors.put(WorkerSubprocessError(
                code,
                "fit recovery event could not be read",
            ))
        except Exception:
            errors.put(WorkerSubprocessError(
                ErrorCode.INTERNAL,
                "fit recovery event processing failed internally",
            ))

    def _signal_process_group(self, process, signum: int) -> None:
        try:
            self._signal_group(process.pid, signum)
        except ProcessLookupError:
            pass

    def _abort_spawned_process(self, process, threads) -> None:
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


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("incomplete subprocess frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _valid_progress(value) -> bool:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if not isinstance(key, str):
            return False
        if item is None or isinstance(item, (str, bool, int)):
            continue
        if isinstance(item, float) and math.isfinite(item):
            continue
        return False
    return True
