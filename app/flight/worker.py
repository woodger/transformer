from __future__ import annotations

from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Callable, Sequence

from app.config import PROJECT_ROOT
from app.flight.arrow import validate_prediction_file
from app.flight.constants import ErrorCode, JobState
from app.flight.contract import model_config_to_api, train_config_to_api
from app.flight.errors import ServiceError
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.process import ProcessRecoveryError, capture_worker_process
from app.flight.worker_plan import (
    ExecutionInput,
    ExecutionPlan,
    WorkerPlanBuilder,
    WorkerPlanError,
)
from app.storage.checkpoint import CHECKPOINT_FORMAT, load_checkpoint_metadata
from app.training.run_config import ModelConfig, TrainConfig


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
class _Failure(Exception):
    code: ErrorCode
    message: str
    exit_code: int | None = None

    def __post_init__(self):
        Exception.__init__(self, self.message)


@dataclass
class _ActiveAttempt:
    job_id: str
    attempt: int
    cancel: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None


@dataclass(frozen=True)
class _ProcessResult:
    exit_code: int
    stderr_tail: bytes
    outputs: tuple[dict, ...] = ()


class WorkerPool:
    """Durable single-instance worker pool for queued Flight jobs.

    PostgreSQL records lifecycle transitions; in-process FIFO queues wake the
    CPU and CUDA lanes without polling the database. Each claimed job gets
    exactly one legacy CLI subprocess.
    """

    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
        popen_factory: Callable = subprocess.Popen,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
        signal_group: Callable[[int, int], None] = os.killpg,
        python_executable: str | None = None,
        cli_path: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if config.cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than zero")
        if config.cuda_capacity != 1:
            raise ValueError("Flight contract v1 requires exactly one CUDA lane")
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.logger = logger or JsonLogger()
        self.metrics = metrics or OperationalMetrics()
        self._popen = popen_factory
        self._argv_hook = argv_hook
        self._signal_group = signal_group
        self._python = python_executable or sys.executable
        self._cli_path = cli_path or os.path.join(PROJECT_ROOT, "app", "main.py")
        self._monotonic = monotonic
        self._plan_builder = WorkerPlanBuilder(
            config,
            ledger,
            spool,
            python_executable=self._python,
            cli_path=self._cli_path,
        )

        self._lifecycle_lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active: dict[str, _ActiveAttempt] = {}
        self._pending_cancellations: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._queues: dict[str, queue.Queue[str | None]] = {
            "cpu": queue.Queue(),
            "cuda": queue.Queue(),
        }
        self._queue_lock = threading.Lock()
        self._enqueued: set[str] = set()
        self._stop_claiming = threading.Event()
        self._force_stop = threading.Event()
        self._started = False

    @property
    def lane_counts(self) -> dict[str, int]:
        return {"cpu": self.config.cpu_capacity, "cuda": 1}

    def start(self) -> "WorkerPool":
        with self._lifecycle_lock:
            if self._started:
                return self
            self._started = True
            for job in self.ledger.queued_jobs():
                self._enqueue(job)
            for device, count in self.lane_counts.items():
                for index in range(count):
                    worker_id = f"{os.getpid()}-{device}-{index}"
                    thread = threading.Thread(
                        target=self._lane,
                        args=(device, worker_id),
                        name=f"transformer-flight-{device}-{index}",
                        daemon=True,
                    )
                    self._threads.append(thread)
                    thread.start()
        return self

    def notify_queued(self, job_id: str | None = None) -> None:
        """Enqueue a job after the coordinator commits QUEUED."""
        if job_id is None:
            for job in self.ledger.queued_jobs():
                self._enqueue(job)
            return
        job = self.ledger.get_job(job_id)
        if job is not None:
            self._enqueue(job)

    def notify_cancel(self, job_id: str) -> None:
        """Cancellation callback used after RUNNING -> CANCELLING commits."""
        with self._active_lock:
            active = self._active.get(job_id)
            if active is None:
                self._pending_cancellations.add(job_id)
            else:
                active.cancel.set()

    def stop_claiming(self) -> None:
        """Close the durable queue-claim boundary without cancelling work."""
        with self._claim_lock:
            if self._stop_claiming.is_set():
                return
            self._stop_claiming.set()
            for device, count in self.lane_counts.items():
                for _ in range(count):
                    self._queues[device].put(None)

    def shutdown(self, timeout: float | None = None) -> None:
        self.stop_claiming()
        drain = self.config.shutdown_drain_seconds if timeout is None else max(0.0, timeout)
        deadline = self._monotonic() + drain
        for thread in tuple(self._threads):
            remaining = max(0.0, deadline - self._monotonic())
            thread.join(remaining)
        if any(thread.is_alive() for thread in self._threads):
            self._force_stop.set()
            with self._active_lock:
                for active in self._active.values():
                    active.cancel.set()
            for thread in tuple(self._threads):
                thread.join(self.config.cancel_grace_seconds + 1.0)
        if any(thread.is_alive() for thread in self._threads):
            self.logger.event(
                "flight.worker.drain_exceeded",
                activeWorkers=sum(thread.is_alive() for thread in self._threads),
            )
            # The process-level state lock cannot be released while a worker
            # thread can still mutate PostgreSQL or the spool. Process groups have
            # already received SIGKILL; finish joining before returning control
            # to FlightApplication.shutdown().
            for thread in tuple(self._threads):
                thread.join()

    def run_once(self, device: str, *, worker_id: str | None = None) -> bool:
        """Atomically claim and execute at most one job for deterministic tests."""
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self._stop_claiming.is_set():
            return False
        job_id = self._dequeue(device)
        if job_id is None:
            for job in self.ledger.queued_jobs():
                self._enqueue(job)
            job_id = self._dequeue(device)
        if job_id is None:
            return False
        return self._claim_and_execute(
            device,
            job_id,
            worker_id or f"{os.getpid()}-{device}-manual",
        )

    def _claim_and_execute(self, device: str, job_id: str, worker_id: str) -> bool:
        with self._claim_lock:
            if self._stop_claiming.is_set():
                return False
            claimed = self.ledger.claim_job(
                job_id,
                device,
                worker_id=worker_id,
            )
        if claimed is None:
            return False
        queue_wait = max(
            0.0,
            float(claimed["started_at"]) - float(claimed["queued_at"]),
        )
        self.metrics.add("jobsStarted")
        self.metrics.add("workerQueueWaitSeconds", queue_wait)
        self.metrics.record_transition(
            JobState.QUEUED.value,
            JobState.RUNNING.value,
        )
        self.logger.event(
            "flight.job.transition",
            jobId=claimed["job_id"],
            attempt=claimed["attempt"],
            device=device,
            fromState=JobState.QUEUED.value,
            toState=JobState.RUNNING.value,
            queueWaitSeconds=queue_wait,
        )
        self._execute_claimed(claimed)
        return True

    def build_argv(self, job: dict, attempt: int | None = None) -> list[str]:
        """Render the trusted CLI invocation; exposed for contract-level tests."""
        attempt = job["attempt"] if attempt is None else attempt
        try:
            return list(self._plan_builder.build_argv(
                job,
                attempt,
                argv_hook=self._argv_hook,
            ))
        except WorkerPlanError as exc:
            raise _Failure(exc.code, exc.message) from exc

    def _lane(self, device: str, worker_id: str) -> None:
        while True:
            job_id = self._queues[device].get()
            if job_id is None:
                return
            with self._queue_lock:
                self._enqueued.discard(job_id)
            try:
                self._claim_and_execute(device, job_id, worker_id)
            except Exception as exc:
                # A single unexpected attempt-finalization/storage error must
                # not permanently remove capacity from the service.  The
                # claimed job is never requeued here (especially not fit); its
                # durable state is left for explicit recovery/reconciliation.
                self.metrics.add("workerLaneErrors")
                self.logger.event(
                    "flight.worker.lane_error",
                    device=device,
                    workerId=worker_id,
                    errorType=type(exc).__name__,
                )

    def _enqueue(self, job: dict) -> None:
        if job.get("state") != JobState.QUEUED.value:
            return
        device = job.get("selected_device")
        if device not in self._queues:
            return
        job_id = job["job_id"]
        with self._queue_lock:
            if job_id in self._enqueued or self._stop_claiming.is_set():
                return
            self._enqueued.add(job_id)
            self._queues[device].put(job_id)

    def _dequeue(self, device: str) -> str | None:
        try:
            job_id = self._queues[device].get_nowait()
        except queue.Empty:
            return None
        if job_id is None:
            return None
        with self._queue_lock:
            self._enqueued.discard(job_id)
        return job_id

    def _execute_claimed(self, job: dict) -> None:
        job_id = job["job_id"]
        attempt = job["attempt"]
        active = _ActiveAttempt(job_id, attempt)
        with self._active_lock:
            self._active[job_id] = active
            if job_id in self._pending_cancellations:
                self._pending_cancellations.remove(job_id)
                active.cancel.set()

        started = self._monotonic()
        succeeded = False
        try:
            current = self.ledger.get_job(job_id)
            if current is None:
                raise _Failure(ErrorCode.INTERNAL, "claimed job disappeared")
            if current["state"] == JobState.CANCELLING.value:
                active.cancel.set()
                raise _Failure(ErrorCode.CANCELLED, "job cancellation was requested")

            self.spool.ensure_free_space(self.config.disk_min_free_bytes)
            try:
                plan = self._plan_builder.build(
                    job,
                    attempt,
                    argv_hook=self._argv_hook,
                )
            except WorkerPlanError as exc:
                raise _Failure(exc.code, exc.message) from exc
            result = self._run_process(job, plan, active)
            if job["operation"] == "predict":
                self._publish_outputs(job, plan.inputs, result.outputs)
            else:
                self._publish_model(job)
            succeeded = True
            self.metrics.add("jobsSucceeded")
        except BaseException as exc:
            current = self.ledger.get_job(job_id)
            if current is not None and current["state"] == JobState.SUCCEEDED.value:
                succeeded = True
            else:
                self._cleanup_unpublished(job)
                failure = self._coerce_failure(exc)
                if failure.code == ErrorCode.CUDA_OUT_OF_MEMORY:
                    self.metrics.add("cudaOutOfMemory")
                    self.logger.event(
                        "flight.cuda.oom",
                        jobId=job_id,
                        attempt=attempt,
                        device=job["selected_device"],
                    )
                elif failure.code == ErrorCode.DEVICE_UNAVAILABLE:
                    self.metrics.add("cudaUnavailableDuringExecution")
                # Cleanup can race with a committed cancel action.  Refresh
                # state before choosing the terminal outcome so cancel wins
                # whenever it was registered before final artifact commit.
                current = self.ledger.get_job(job_id)
                if (
                    current is not None
                    and current["state"] == JobState.CANCELLING.value
                ) or failure.code == ErrorCode.CANCELLED:
                    self._finish_cancelled(job, failure.exit_code)
                    self.metrics.add("jobsCancelled")
                else:
                    self._finish_failed(job, failure)
                    self.metrics.add("jobsFailed")
                self.logger.event(
                    "flight.worker.failed",
                    jobId=job_id,
                    attempt=attempt,
                    code=failure.code.value,
                    errorType=type(exc).__name__,
                )
        finally:
            with self._active_lock:
                self._active.pop(job_id, None)
                self._pending_cancellations.discard(job_id)
            elapsed = self._monotonic() - started
            self.metrics.add("workerRunSeconds", elapsed)
            self.logger.event(
                "flight.worker.finished",
                jobId=job_id,
                attempt=attempt,
                device=job["selected_device"],
                succeeded=succeeded,
                runSeconds=elapsed,
            )

    def _run_process(
        self,
        job: dict,
        plan: ExecutionPlan,
        active: _ActiveAttempt,
    ) -> _ProcessResult:
        job_id = job["job_id"]
        attempt = job["attempt"]
        inputs = plan.inputs
        argv = plan.argv
        stdout_path = self.spool.attempt_stdout_path(job_id, attempt)
        stderr_path = self.spool.attempt_stderr_path(job_id, attempt)
        self.spool.ensure_parent(stdout_path)
        self.spool.ensure_parent(stderr_path)

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        supervised_argv = [
            self._python,
            os.path.join(PROJECT_ROOT, "app", "flight", "process_supervisor.py"),
            str(os.getpid()),
            "--",
            *argv,
        ]
        uses_spooled_fit = plan.uses_spooled_fit
        try:
            process = self._popen(
                supervised_argv,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL if uses_spooled_fit else subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                close_fds=True,
                bufsize=0,
            )
        except OSError as exc:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "worker subprocess could not be started",
            ) from exc
        active.process = process
        deadline = self._monotonic() + self.config.subprocess_timeout_seconds
        errors: queue.Queue[_Failure] = queue.Queue()
        finished = threading.Event()
        stderr_tail: list[bytes] = []
        outputs: list[dict] = []
        threads: list[threading.Thread] = []
        started_threads: list[threading.Thread] = []
        try:
            try:
                identity = capture_worker_process(process.pid)
            except ProcessRecoveryError as exc:
                raise _Failure(
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
                device=job["selected_device"],
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
            if not uses_spooled_fit:
                threads.insert(
                    0,
                    threading.Thread(
                        target=self._feed_frames,
                        args=(process.stdin, inputs, errors),
                        name=f"flight-stdin-{job_id}",
                        daemon=True,
                    ),
                )
            if job["operation"] == "predict":
                # Keep a deterministic empty text log; binary stdout is staged as Arrow.
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
                threads.append(threading.Thread(
                    target=self._tail_metrics,
                    args=(job, inputs, finished, errors),
                    name=f"flight-metrics-{job_id}",
                    daemon=True,
                ))
            for thread in threads:
                thread.start()
                started_threads.append(thread)
        except BaseException:
            finished.set()
            self._abort_spawned_process(process, started_threads)
            active.process = None
            raise

        failure: _Failure | None = None
        term_sent_at: float | None = None
        killed = False
        try:
            while process.poll() is None:
                now = self._monotonic()
                if failure is None and active.cancel.is_set():
                    code = (
                        ErrorCode.EXECUTION_INTERRUPTED
                        if self._force_stop.is_set()
                        else ErrorCode.CANCELLED
                    )
                    message = (
                        "worker execution was interrupted by service shutdown"
                        if code == ErrorCode.EXECUTION_INTERRUPTED
                        else "job cancellation was requested"
                    )
                    failure = _Failure(code, message)
                if failure is None:
                    try:
                        failure = errors.get_nowait()
                    except queue.Empty:
                        pass
                if failure is None and now >= deadline:
                    failure = _Failure(
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
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except OSError:
                pass
            # A descendant can inherit stdout/stderr and keep the pipe open
            # after the direct child exits.  Continue enforcing the same
            # execution deadline while pumps drain, and signal the whole
            # process group even when its original leader has already exited.
            while any(thread.is_alive() for thread in started_threads):
                now = self._monotonic()
                if failure is None and active.cancel.is_set():
                    code = (
                        ErrorCode.EXECUTION_INTERRUPTED
                        if self._force_stop.is_set()
                        else ErrorCode.CANCELLED
                    )
                    message = (
                        "worker execution was interrupted by service shutdown"
                        if code == ErrorCode.EXECUTION_INTERRUPTED
                        else "job cancellation was requested"
                    )
                    failure = _Failure(code, message)
                if failure is None:
                    try:
                        failure = errors.get_nowait()
                    except queue.Empty:
                        pass
                if failure is None and now >= deadline:
                    failure = _Failure(
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
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except OSError:
                    pass
            active.process = None

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
        ):
            raise classified_exit
        if failure is not None:
            raise _Failure(failure.code, failure.message, exit_code)
        if classified_exit is not None:
            raise classified_exit
        return _ProcessResult(exit_code, tail, tuple(outputs))

    def _feed_frames(
        self,
        stream,
        inputs: tuple[ExecutionInput, ...],
        errors,
    ) -> None:
        try:
            for item in inputs:
                size = item.byte_count
                stream.write(size.to_bytes(_FRAME_HEADER_BYTES, "big", signed=False))
                with open(item.absolute_path, "rb") as source:
                    shutil.copyfileobj(source, stream, _COPY_CHUNK_BYTES)
                stream.flush()
            stream.write((0).to_bytes(_FRAME_HEADER_BYTES, "big", signed=False))
            stream.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            errors.put(_Failure(
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
            errors.put(_Failure(code, "worker log could not be persisted"))
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
        job: dict,
        inputs: tuple[ExecutionInput, ...],
        outputs: list[dict],
        errors,
    ) -> None:
        try:
            for item in inputs:
                header = _read_exact(stream, _FRAME_HEADER_BYTES)
                size = int.from_bytes(header, "big", signed=False)
                if size <= 0:
                    raise _Failure(
                        ErrorCode.MALFORMED_OUTPUT,
                        "prediction subprocess emitted a zero-length frame",
                    )
                if size > self.config.max_payload_bytes:
                    raise _Failure(
                        ErrorCode.MALFORMED_OUTPUT,
                        "prediction subprocess output exceeded the payload limit",
                    )
                destination = self.spool.attempt_output_path(
                    job["job_id"],
                    job["attempt"],
                    item.ordinal,
                )
                output = self._stage_prediction(
                    stream,
                    destination,
                    size,
                    job["prediction_column"],
                    item.rows,
                    item.ordinal,
                )
                outputs.append(output)
            if stream.read(1) != b"":
                raise _Failure(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction subprocess emitted more frames than expected",
                )
        except _Failure as exc:
            errors.put(exc)
        except OSError as exc:
            if exc.errno in _DISK_FULL_ERRNOS:
                errors.put(_Failure(
                    ErrorCode.DISK_FULL,
                    "prediction output could not be persisted",
                ))
            else:
                errors.put(_Failure(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction subprocess emitted malformed Arrow output",
                ))
        except (EOFError, ServiceError, ValueError) as exc:
            errors.put(_Failure(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess emitted malformed Arrow output",
            ))

    def _stage_prediction(
        self,
        stream,
        destination: str,
        size: int,
        prediction_column: str,
        expected_rows: int,
        ordinal: int,
    ) -> dict:
        target, temporary = self.spool.create_temporary(destination)
        digest = hashlib.sha256()
        try:
            remaining = size
            while remaining:
                chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
                if not chunk:
                    raise EOFError("incomplete prediction frame")
                target.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
            target.close()
            stats = validate_prediction_file(
                temporary,
                prediction_column,
                expected_rows,
            )
            self.spool.durable_replace(temporary, destination)
            temporary = None
            return {
                "ordinal": ordinal,
                "rows": stats.rows,
                "batches": stats.batches,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "schema_fingerprint": stats.schema_fingerprint,
                "relative_path": self.spool.relative_path(destination),
            }
        finally:
            if not target.closed:
                target.close()
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _tail_metrics(
        self,
        job: dict,
        inputs: tuple[ExecutionInput, ...],
        finished,
        errors,
    ) -> None:
        path = self.spool.attempt_metrics_path(job["job_id"], job["attempt"])
        offset = 0
        inode = None
        partial = b""
        records = 0
        try:
            while True:
                # A process exit makes its metrics file stable, but the waiter
                # may set ``finished`` while this iteration is between stat()
                # and read().  Only finalize when the process was already
                # finished before the iteration began; otherwise perform one
                # final pass over the now-stable file.
                finished_at_iteration_start = finished.is_set()
                try:
                    stat = os.stat(path)
                except FileNotFoundError:
                    if finished_at_iteration_start:
                        raise _Failure(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess did not produce metrics JSONL",
                        )
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
                        raise _Failure(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess emitted malformed metrics JSONL",
                        ) from exc
                    if not _valid_progress(progress):
                        raise _Failure(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess emitted malformed metrics JSONL",
                        )
                    frame = progress.get("frame")
                    if isinstance(frame, int) and 1 <= frame <= len(inputs):
                        progress["ordinal"] = inputs[frame - 1].ordinal
                    self.ledger.update_progress(job["job_id"], progress)
                    records += 1
                if finished_at_iteration_start:
                    if partial.strip():
                        raise _Failure(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess left an incomplete metrics record",
                        )
                    if records == 0:
                        raise _Failure(
                            ErrorCode.MALFORMED_OUTPUT,
                            "fit subprocess produced no metrics records",
                        )
                    return
                finished.wait(0.05)
        except _Failure as exc:
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
            errors.put(_Failure(code, "fit progress could not be read"))
        except Exception:
            errors.put(_Failure(
                ErrorCode.INTERNAL,
                "fit progress processing failed internally",
            ))

    def _publish_outputs(
        self,
        job: dict,
        inputs: tuple[ExecutionInput, ...],
        outputs: tuple[dict, ...],
    ) -> None:
        if len(outputs) != len(inputs):
            raise _Failure(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess output count did not match input count",
            )
        result = {
            "outputs": [
                {"ordinal": item["ordinal"], "rows": item["rows"]}
                for item in outputs
            ]
        }
        self.ledger.publish_outputs(
            job["job_id"],
            job["attempt"],
            outputs,
            result=result,
        )
        for item in outputs:
            self.metrics.add("predictionOutputBytes", item["bytes"])
            self.metrics.add("predictionOutputRows", item["rows"])
            self.metrics.add("predictionOutputBatches", item["batches"])
            self.logger.event(
                "flight.output.published",
                jobId=job["job_id"],
                ordinal=item["ordinal"],
                rows=item["rows"],
                batches=item["batches"],
                bytes=item["bytes"],
            )
        self._record_transition(
            job,
            JobState.RUNNING.value,
            JobState.SUCCEEDED.value,
        )

    def _publish_model(self, job: dict) -> None:
        attempt_path = self.spool.attempt_checkpoint_path(
            job["job_id"], job["attempt"]
        )
        if not os.path.isfile(attempt_path):
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess did not create a checkpoint",
            )
        try:
            checkpoint = load_checkpoint_metadata(attempt_path, "cpu")
            actual_model = ModelConfig.from_dict(checkpoint.get("model_config"))
            actual_train = TrainConfig.from_dict(checkpoint.get("train_config"))
        except Exception as exc:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        expected_model = ModelConfig.from_dict(job["model_config"])
        expected_train = TrainConfig.from_dict(job["training_config"])
        if checkpoint.get("format") != CHECKPOINT_FORMAT or actual_model is None:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an unsupported checkpoint",
            )
        expected_values = expected_model.to_dict()
        expected_values["feature_dim"] = job.get("feature_dim")
        if actual_model.to_dict() != expected_values or actual_train != expected_train:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit checkpoint configuration differs from the immutable job config",
            )

        digest = _sha256_file(attempt_path)
        byte_count = os.path.getsize(attempt_path)
        model_ref = f"mdl_{uuid.uuid4().hex}"
        model_directory = self.spool.model_directory(model_ref)
        checkpoint_path = self.spool.model_checkpoint_path(model_ref)
        metadata_path = self.spool.model_metadata_path(model_ref)
        try:
            safe_checkpoint = {
                "format": checkpoint["format"],
                "serviceVersion": checkpoint.get("version"),
                "sha256": digest,
                "bytes": byte_count,
                "modelConfig": model_config_to_api(actual_model),
                "trainConfig": train_config_to_api(actual_train),
                "dataSchema": _data_schema_to_api(
                    checkpoint.get("data_schema"),
                    actual_model,
                ),
                "checkpointSelection": _checkpoint_selection_to_api(
                    (checkpoint.get("extra") or {}).get("checkpoint_selection"),
                    actual_train,
                ),
            }
        except _Failure:
            raise
        except Exception as exc:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        if not isinstance(safe_checkpoint["serviceVersion"], str) or not safe_checkpoint[
            "serviceVersion"
        ]:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit checkpoint does not contain a service version",
            )
        metadata = {
            "modelRef": model_ref,
            "label": job["model_label"],
            # Internal snake_case copies let predict create resolve a generation
            # without depending on the public status document representation.
            "model_config": actual_model.to_dict(),
            "train_config": actual_train.to_dict(),
            "checkpoint": safe_checkpoint,
        }
        try:
            with open(attempt_path, "rb") as source:
                with self.spool.staged_file(checkpoint_path) as (target, _):
                    shutil.copyfileobj(source, target, _COPY_CHUNK_BYTES)
            self.spool.atomic_write_json(metadata_path, metadata)
            self.ledger.publish_model(
                job["job_id"],
                job["attempt"],
                model_ref=model_ref,
                label=job["model_label"],
                generation=None,
                checkpoint_path=self.spool.model_relative_path(checkpoint_path),
                metadata_path=self.spool.model_relative_path(metadata_path),
                sha256=digest,
                metadata=metadata,
                result={"modelRef": model_ref, "checkpoint": safe_checkpoint},
            )
            self.metrics.add("checkpointBytes", byte_count)
            self.logger.event(
                "flight.model.published",
                jobId=job["job_id"],
                modelRef=model_ref,
                bytes=byte_count,
                sha256=digest,
            )
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.SUCCEEDED.value,
            )
        except BaseException:
            current = self.ledger.get_job(job["job_id"])
            if current is None or current["state"] != JobState.SUCCEEDED.value:
                self.spool.remove(model_directory)
            raise

    def _cleanup_unpublished(self, job: dict) -> None:
        try:
            inputs = self.ledger.list_inputs(job["job_id"])
        except Exception as exc:
            self.logger.event(
                "flight.worker.cleanup_failed",
                jobId=job["job_id"],
                attempt=job["attempt"],
                artifact="attempt",
                errorType=type(exc).__name__,
            )
            return
        paths = [
            self.spool.attempt_output_path(
                job["job_id"], job["attempt"], item["ordinal"]
            )
            for item in inputs
        ]
        paths.append(
            self.spool.attempt_checkpoint_path(job["job_id"], job["attempt"])
        )
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                self.spool.remove(path)
            except Exception as exc:
                # Publication is ledger-gated.  A filesystem cleanup failure
                # can leave only an invisible orphan, which startup
                # reconciliation removes; it must not strand the job RUNNING
                # or kill a worker lane.
                self.logger.event(
                    "flight.worker.cleanup_failed",
                    jobId=job["job_id"],
                    attempt=job["attempt"],
                    artifact="attempt",
                    errorType=type(exc).__name__,
                )

    def _finish_cancelled(self, job: dict, exit_code: int | None) -> None:
        current = self.ledger.get_job(job["job_id"])
        if current is None or current["state"] in (
            JobState.CANCELLED.value,
            JobState.SUCCEEDED.value,
            JobState.FAILED.value,
        ):
            return
        if current["state"] == JobState.RUNNING.value:
            self.ledger.transition_job(job["job_id"], JobState.CANCELLING)
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.CANCELLING.value,
            )
        self.ledger.finish_attempt(
            job["job_id"],
            job["attempt"],
            JobState.CANCELLED,
            exit_code=exit_code,
        )
        self._record_transition(
            job,
            JobState.CANCELLING.value,
            JobState.CANCELLED.value,
        )

    def _finish_failed(self, job: dict, failure: _Failure) -> None:
        current = self.ledger.get_job(job["job_id"])
        if current is None or current["state"] in (
            JobState.SUCCEEDED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        ):
            return
        if current["state"] == JobState.CANCELLING.value:
            self._finish_cancelled(job, failure.exit_code)
            return
        try:
            self.ledger.finish_attempt(
                job["job_id"],
                job["attempt"],
                JobState.FAILED,
                error_code=failure.code,
                error_message=failure.message,
                exit_code=failure.exit_code,
            )
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.FAILED.value,
                code=failure.code.value,
            )
        except (ServiceError, ValueError):
            # If cancel committed its PostgreSQL transition first, the attempted
            # RUNNING -> FAILED transition observes CANCELLING.  Re-read and
            # complete cancellation; otherwise preserve the original error.
            latest = self.ledger.get_job(job["job_id"])
            if latest is not None and latest["state"] == JobState.CANCELLING.value:
                self._finish_cancelled(job, failure.exit_code)
                return
            if latest is not None and latest["state"] in (
                JobState.SUCCEEDED.value,
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            ):
                return
            raise

    def _record_transition(
        self,
        job: dict,
        from_state: str,
        to_state: str,
        *,
        code: str | None = None,
    ) -> None:
        self.metrics.record_transition(from_state, to_state)
        fields = {
            "jobId": job["job_id"],
            "attempt": job["attempt"],
            "device": job["selected_device"],
            "fromState": from_state,
            "toState": to_state,
        }
        if code is not None:
            fields["code"] = code
        self.logger.event("flight.job.transition", **fields)

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
            process.wait(timeout=max(1.0, self.config.cancel_grace_seconds))
        except subprocess.TimeoutExpired:
            self._signal_process_group(process, signal.SIGKILL)
            process.wait()
        for thread in threads:
            thread.join()

    @staticmethod
    def _classify_subprocess_exit(exit_code: int, stderr_tail: bytes) -> _Failure:
        text = stderr_tail.lower()
        if b"cuda" in text and (
            b"out of memory" in text or b"outofmemoryerror" in text
        ):
            return _Failure(
                ErrorCode.CUDA_OUT_OF_MEMORY,
                "CUDA execution ran out of memory",
                exit_code,
            )
        if b"cuda" in text and any(fragment in text for fragment in (
            b"not available",
            b"driver shutting down",
            b"no cuda gpus",
        )):
            return _Failure(
                ErrorCode.DEVICE_UNAVAILABLE,
                "CUDA device became unavailable during execution",
                exit_code,
            )
        return _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            f"worker subprocess exited with status {exit_code}",
            exit_code,
        )

    @staticmethod
    def _coerce_failure(exc: BaseException) -> _Failure:
        if isinstance(exc, _Failure):
            return exc
        if isinstance(exc, ServiceError):
            return _Failure(exc.code, exc.message)
        if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
            return _Failure(ErrorCode.DISK_FULL, "runtime directory is full")
        return _Failure(ErrorCode.INTERNAL, "worker execution failed internally")


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


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _data_schema_to_api(value: dict | None, model_config: ModelConfig) -> dict:
    if not isinstance(value, dict):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint does not contain data schema metadata",
        )
    source = value.get("src")
    target = value.get("tgt")
    missing = value.get("missing")
    if not all(isinstance(item, dict) for item in (source, target, missing)):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid data schema metadata",
        )
    result = {
        "schemaVersion": value.get("schema_version"),
        "tensorDtype": value.get("tensor_dtype"),
        "source": {
            "column": source.get("column"),
            "acceptedElementTypes": source.get("accepted_element_types"),
            "width": source.get("width"),
        },
        "target": {
            "column": target.get("column"),
            "acceptedElementTypes": target.get("accepted_element_types"),
            "width": target.get("width"),
        },
        "featureDim": value.get("feature_dim"),
        "modelInputFeatureDim": value.get("model_input_feature_dim"),
        "contextMode": value.get("context_mode"),
        "normalization": value.get("normalization"),
        "missing": {
            "nanFill": missing.get("nan_fill"),
            "flags": missing.get("flags"),
        },
    }
    expected_types = ["float32", "float64"]
    expected_input_dim = (
        model_config.feature_dim * 2
        if model_config.context_mode == "relaxed"
        else model_config.feature_dim
    )
    if (
        result["schemaVersion"] != 1
        or result["tensorDtype"] != "float32"
        or result["source"]["column"] != "src"
        or result["source"]["acceptedElementTypes"] != expected_types
        or type(result["source"]["width"]) is not int
        or result["source"]["width"] <= 0
        or result["target"]["column"] != "tgt"
        or result["target"]["acceptedElementTypes"] != expected_types
        or result["target"]["width"] != 6
        or result["featureDim"] != model_config.feature_dim
        or result["source"]["width"]
        != model_config.seq_len * model_config.feature_dim
        or result["modelInputFeatureDim"] != expected_input_dim
        or result["contextMode"] != model_config.context_mode
        or result["normalization"] is not None
        or result["missing"]["nanFill"] != 0.0
        or result["missing"]["flags"]
        != ("per-feature" if model_config.context_mode == "relaxed" else "none")
    ):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid data schema metadata",
        )
    return result


def _checkpoint_selection_to_api(
    value: dict | None,
    train_config: TrainConfig,
) -> dict:
    value = value if isinstance(value, dict) else {}
    result = {
        "monitor": value.get("monitor", train_config.monitor),
        "monitorMinImprovement": value.get(
            "monitor_min_improvement",
            train_config.monitor_min_improvement,
        ),
        "bestMonitor": value.get("best_monitor"),
        "bestFrame": value.get("best_frame"),
        "bestEpoch": value.get("best_epoch"),
        "baselinePassed": bool(value.get("baseline_passed", False)),
        "source": value.get("source", "current"),
    }
    best_monitor = result["bestMonitor"]
    best_frame = result["bestFrame"]
    best_epoch = result["bestEpoch"]
    if (
        result["monitor"] != train_config.monitor
        or result["monitorMinImprovement"] != train_config.monitor_min_improvement
        or (
            best_monitor is not None
            and (
                isinstance(best_monitor, bool)
                or not isinstance(best_monitor, (int, float))
                or not math.isfinite(best_monitor)
            )
        )
        or (
            best_frame is not None
            and (type(best_frame) is not int or best_frame < 0)
        )
        or (
            best_epoch is not None
            and (type(best_epoch) is not int or best_epoch < 0)
        )
        or (
            "baseline_passed" in value
            and not isinstance(value["baseline_passed"], bool)
        )
        or result["source"] not in ("best_monitor", "current")
    ):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        )
    return result
