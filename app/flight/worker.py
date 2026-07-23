from __future__ import annotations

from dataclasses import dataclass, field
import errno
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence

from app.config import PROJECT_ROOT
from app.flight.constants import ErrorCode, JobState
from app.flight.errors import ServiceError
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.worker_artifacts import (
    WorkerArtifactError,
    WorkerArtifactPublisher,
)
from app.flight.worker_plan import (
    WorkerPlanBuilder,
    WorkerPlanError,
)
from app.flight.worker_subprocess import (
    WorkerSubprocessError,
    WorkerSubprocessRunner,
)


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
        self._artifact_publisher = WorkerArtifactPublisher(
            ledger,
            spool,
            logger=self.logger,
            metrics=self.metrics,
        )
        self._subprocess_runner = WorkerSubprocessRunner(
            config,
            ledger,
            spool,
            logger=self.logger,
            popen_factory=self._popen,
            signal_group=self._signal_group,
            python_executable=self._python,
            stage_prediction=self._artifact_publisher.stage_prediction,
            monotonic=self._monotonic,
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
            try:
                result = self._subprocess_runner.run(
                    job,
                    plan,
                    cancel=active.cancel,
                    force_stop=self._force_stop,
                )
            except WorkerSubprocessError as exc:
                raise _Failure(
                    exc.code,
                    exc.message,
                    exc.exit_code,
                ) from exc
            try:
                if job["operation"] == "predict":
                    self._artifact_publisher.publish_outputs(
                        job,
                        plan.inputs,
                        result.outputs,
                    )
                else:
                    self._artifact_publisher.publish_model(job)
            except WorkerArtifactError as exc:
                raise _Failure(exc.code, exc.message) from exc
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.SUCCEEDED.value,
            )
            succeeded = True
            self.metrics.add("jobsSucceeded")
        except BaseException as exc:
            current = self.ledger.get_job(job_id)
            if current is not None and current["state"] == JobState.SUCCEEDED.value:
                succeeded = True
            else:
                self._artifact_publisher.cleanup_unpublished(job)
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

    @staticmethod
    def _coerce_failure(exc: BaseException) -> _Failure:
        if isinstance(exc, _Failure):
            return exc
        if isinstance(exc, ServiceError):
            return _Failure(exc.code, exc.message)
        if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
            return _Failure(ErrorCode.DISK_FULL, "runtime directory is full")
        return _Failure(ErrorCode.INTERNAL, "worker execution failed internally")
