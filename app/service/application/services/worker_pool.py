from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable

from app.service.application.ports.devices import DeviceLeaseManager
from app.service.application.ports.jobs import JobRepository
from app.service.application.ports.workers import WorkerExecutor
from app.service.domain.job import ExecutionState
from app.service.domain.records import ExecutionJobRecord


class WorkerPool:
    """Durable single-instance worker pool for queued Flight jobs.

    PostgreSQL records lifecycle transitions. In-process FIFO notifications
    wake CPU and CUDA lanes; the maintenance service periodically calls
    ``notify_queued()`` to recover a notification lost after commit. Each
    claimed attempt gets exactly one trusted CLI subprocess.
    """

    def __init__(
        self,
        config,
        ledger: JobRepository,
        *,
        logger,
        metrics,
        device_inventory: DeviceLeaseManager,
        attempt_executor: WorkerExecutor | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if config.cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than zero")
        self.config = config
        self.ledger = ledger
        self.logger = logger
        self.metrics = metrics
        self._monotonic = monotonic
        self.device_inventory = device_inventory
        self._attempt_executor = attempt_executor

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

    def attach_attempt_executor(
        self,
        attempt_executor: WorkerExecutor,
    ) -> None:
        if self._started or self._attempt_executor is not None:
            raise RuntimeError("worker attempt executor is already configured")
        self._attempt_executor = attempt_executor

    def _executor(self) -> WorkerExecutor:
        if self._attempt_executor is None:
            raise RuntimeError("worker attempt executor is not configured")
        return self._attempt_executor

    @property
    def lane_counts(self) -> dict[str, int]:
        return {
            "cpu": self.config.cpu_capacity,
            "cuda": len(
                self.device_inventory.schedulable_devices()
            ),
        }

    def start(self) -> WorkerPool:
        with self._lifecycle_lock:
            if self._started:
                return self
            self._started = True
            for job in self.ledger.queued_execution_jobs():
                self._enqueue(job)
            for index in range(self.config.cpu_capacity):
                worker_id = f"{os.getpid()}-cpu-{index}"
                thread = threading.Thread(
                    target=self._lane,
                    args=("cpu", worker_id),
                    name=f"transformer-flight-cpu-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
            for cuda_device in (
                self.device_inventory.schedulable_devices()
            ):
                worker_id = (
                    f"{os.getpid()}-cuda-{cuda_device.ordinal}"
                )
                thread = threading.Thread(
                    target=self._lane,
                    args=("cuda", worker_id, cuda_device.device_id),
                    name=(
                        "transformer-flight-cuda-"
                        f"{cuda_device.ordinal}"
                    ),
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
        self._executor().notify_cancel(job_id)

    def notify_input(self, job_id: str) -> None:
        """Wake the bounded worker control channel after a durable commit."""

        self._executor().notify_input(job_id)

    def stop_claiming(self) -> None:
        """Close the durable queue-claim boundary without cancelling work."""
        with self._claim_lock:
            if self._stop_claiming.is_set():
                return
            self._stop_claiming.set()
            for _ in range(self.config.cpu_capacity):
                self._queues["cpu"].put(None)
            for _ in self.device_inventory.snapshot().devices:
                self._queues["cuda"].put(None)

    def shutdown(self, timeout: float | None = None) -> None:
        self.stop_claiming()
        drain = self.config.shutdown_drain_seconds if timeout is None else max(0.0, timeout)
        deadline = self._monotonic() + drain
        for thread in tuple(self._threads):
            remaining = max(0.0, deadline - self._monotonic())
            thread.join(remaining)
        if any(thread.is_alive() for thread in self._threads):
            self._executor().interrupt_for_shutdown()
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
        device_id = None
        if device == "cuda":
            available = self.device_inventory.schedulable_devices()
            if not available:
                self._enqueue_job_id(device, job_id)
                return False
            device_id = available[0].device_id
            if not self.device_inventory.mark_busy(device_id):
                self._enqueue_job_id(device, job_id)
                return False
        try:
            claimed = self._claim_and_execute(
                device,
                job_id,
                worker_id or f"{os.getpid()}-{device}-manual",
                device_id=device_id,
            )
        finally:
            if device_id is not None:
                self.device_inventory.release(device_id)
        if not claimed:
            self._requeue_if_pending(job_id)
        return claimed

    def _claim_and_execute(
        self,
        device: str,
        job_id: str,
        worker_id: str,
        *,
        device_id: str | None = None,
    ) -> bool:
        pending = self.ledger.get_execution_job(job_id)
        from_state = (
            ExecutionState.QUEUED
            if pending is None
            else pending.execution_state
        )
        with self._claim_lock:
            if self._stop_claiming.is_set():
                return False
            claimed = self.ledger.claim_execution_job(
                job_id,
                device,
                worker_id=worker_id,
                device_id=device_id,
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
            from_state.value,
            ExecutionState.RUNNING.value,
        )
        self.logger.event(
            "flight.job.transition",
            jobId=claimed.job_id,
            attempt=claimed.attempt,
            device=device,
            deviceId=device_id,
            fromState=from_state.value,
            toState=ExecutionState.RUNNING.value,
            queueWaitSeconds=queue_wait,
        )
        self._executor().execute(claimed)
        return True

    def _lane(
        self,
        device: str,
        worker_id: str,
        device_id: str | None = None,
    ) -> None:
        while True:
            if (
                device_id is not None
                and self.device_inventory.is_quarantined(device_id)
            ):
                return
            job_id = self._queues[device].get()
            if job_id is None:
                return
            with self._queue_lock:
                self._enqueued.discard(job_id)
            try:
                if (
                    device_id is not None
                    and not self.device_inventory.mark_busy(device_id)
                ):
                    self._enqueue_job_id(device, job_id)
                    if self.device_inventory.is_quarantined(device_id):
                        return
                    continue
                try:
                    claimed = self._claim_and_execute(
                        device,
                        job_id,
                        worker_id,
                        device_id=device_id,
                    )
                finally:
                    if device_id is not None:
                        self.device_inventory.release(device_id)
                if not claimed:
                    self._requeue_if_pending(job_id)
            except Exception as exc:
                # A single unexpected attempt-finalization/storage error must
                # not permanently remove capacity from the service.  The
                # Durable attempt finalization decides whether a fit becomes
                # RETRYING. The lane never invents a retry from an exception.
                self.metrics.add("workerLaneErrors")
                self.logger.event(
                    "flight.worker.lane_error",
                    device=device,
                    workerId=worker_id,
                    errorType=type(exc).__name__,
                )

    def _requeue_if_pending(self, job_id: str) -> None:
        pending = self.ledger.get_execution_job(job_id)
        if pending is not None:
            self._enqueue(pending)

    def _enqueue(self, job: ExecutionJobRecord) -> None:
        if job.execution_state not in (
            ExecutionState.QUEUED,
            ExecutionState.RETRYING,
        ):
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

    def _enqueue_job_id(self, device: str, job_id: str) -> None:
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
