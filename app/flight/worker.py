from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence

from app.config import PROJECT_ROOT
from app.flight.constants import JobState
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.records import (
    ExecutionJobRecord,
    execution_job_from_mapping,
)
from app.flight.worker_artifacts import WorkerArtifactPublisher
from app.flight.worker_attempt import (
    WorkerAttemptError,
    WorkerAttemptExecutor,
)
from app.flight.worker_plan import (
    WorkerPlanBuilder,
    WorkerPlanError,
)
from app.flight.worker_subprocess import WorkerSubprocessRunner


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
        self._attempt_executor = WorkerAttemptExecutor(
            config,
            ledger,
            spool,
            self._plan_builder,
            self._subprocess_runner,
            self._artifact_publisher,
            logger=self.logger,
            metrics=self.metrics,
            argv_hook=self._argv_hook,
            monotonic=self._monotonic,
        )

        self._lifecycle_lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._queues: dict[str, queue.Queue[str | None]] = {
            "cpu": queue.Queue(),
            "cuda": queue.Queue(),
        }
        self._queue_lock = threading.Lock()
        self._enqueued: set[str] = set()
        self._stop_claiming = threading.Event()
        self._started = False

    @property
    def lane_counts(self) -> dict[str, int]:
        return {"cpu": self.config.cpu_capacity, "cuda": 1}

    def start(self) -> WorkerPool:
        with self._lifecycle_lock:
            if self._started:
                return self
            self._started = True
            for job in self.ledger.queued_execution_jobs():
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
            for job in self.ledger.queued_execution_jobs():
                self._enqueue(job)
            return
        job = self.ledger.get_execution_job(job_id)
        if job is not None:
            self._enqueue(job)

    def notify_cancel(self, job_id: str) -> None:
        """Cancellation callback used after RUNNING -> CANCELLING commits."""
        self._attempt_executor.notify_cancel(job_id)

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
            self._attempt_executor.interrupt_for_shutdown()
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
            for job in self.ledger.queued_execution_jobs():
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
            claimed = self.ledger.claim_execution_job(
                job_id,
                device,
                worker_id=worker_id,
            )
        if claimed is None:
            return False
        queue_wait = max(
            0.0,
            float(claimed.started_at) - float(claimed.queued_at),
        )
        self.metrics.add("jobsStarted")
        self.metrics.add("workerQueueWaitSeconds", queue_wait)
        self.metrics.record_transition(
            JobState.QUEUED.value,
            JobState.RUNNING.value,
        )
        self.logger.event(
            "flight.job.transition",
            jobId=claimed.job_id,
            attempt=claimed.attempt,
            device=device,
            fromState=JobState.QUEUED.value,
            toState=JobState.RUNNING.value,
            queueWaitSeconds=queue_wait,
        )
        self._attempt_executor.execute(claimed)
        return True

    def build_argv(self, job: dict, attempt: int | None = None) -> list[str]:
        """Render the trusted CLI invocation; exposed for contract-level tests."""
        attempt = job["attempt"] if attempt is None else attempt
        record = execution_job_from_mapping(job)
        try:
            return list(self._plan_builder.build_argv(
                record,
                attempt,
                argv_hook=self._argv_hook,
                legacy_job=job,
            ))
        except WorkerPlanError as exc:
            raise WorkerAttemptError(exc.code, exc.message) from exc

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

    def _enqueue(self, job: ExecutionJobRecord) -> None:
        if job.state != JobState.QUEUED:
            return
        device = job.selected_device
        if device not in self._queues:
            return
        job_id = job.job_id
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
