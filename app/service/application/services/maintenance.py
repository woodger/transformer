from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from app.service.application.ports.maintenance import (
    MaintenanceArtifactStore,
    MaintenanceRepository,
    RetentionArtifactStore,
)
from app.service.application.ports.models import ModelDeletionRepository
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.operations import DiskUsage


class MaintenanceConfig(Protocol):
    @property
    def retention_seconds(self) -> float: ...

    @property
    def input_idle_timeout_seconds(self) -> float: ...


@dataclass(frozen=True)
class MaintenanceResult:
    now: float
    cutoff: float
    expired_tickets: int
    expired_input_jobs: tuple[str, ...]
    deleted_jobs: tuple[str, ...]
    removed_job_directories: tuple[str, ...]
    missing_job_directories: tuple[str, ...]
    failed_job_directories: tuple[str, ...]
    pending_job_directories: tuple[str, ...]
    completed_model_deletions: tuple[str, ...]
    removed_model_directories: tuple[str, ...]
    missing_model_directories: tuple[str, ...]
    failed_model_directories: tuple[str, ...]


class MaintenanceService:
    """Reconcile durable queues and expire tickets and retained terminal jobs.

    Terminal ledger rows are deleted before their server-owned job directories.
    Reversing that order could leave live ledger references to missing artifacts.
    A process crash between the two operations can therefore leave an orphan job
    directory. Startup reconciliation must remove ``spool/jobs/<jobId>`` entries
    whose canonical job IDs are absent from the ledger-provided known-job set.
    The in-memory pending set only closes retry gaps while this process remains
    alive; it is deliberately not treated as durable state.

    Published model generations outlive the producing job. Their directories
    are removed only after an explicit PostgreSQL deletion request; the model
    identity is archived and the working row is purged only after the
    filesystem operation succeeds.
    """

    def __init__(
        self,
        config: MaintenanceConfig,
        ledger: MaintenanceRepository,
        spool: MaintenanceArtifactStore,
        recovery_store: RetentionArtifactStore | None = None,
        *,
        model_deletions: ModelDeletionRepository | None = None,
        interval_seconds: float = 60.0,
        queue_reconciler: Callable[[], None] | None = None,
        logger: EventLogger,
        metrics: OperationalMetricSink,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a positive finite number")
        if config.retention_seconds <= 0:
            raise ValueError("retention_seconds must be greater than zero")

        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.recovery_store = recovery_store
        self.model_deletions = model_deletions
        self.interval_seconds = float(interval_seconds)
        self._queue_reconciler = queue_reconciler
        self.logger = logger
        self.metrics = metrics

        self._lifecycle_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_job_directories: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> MaintenanceService:
        with self._lifecycle_lock:
            if self._thread is not None:
                return self
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="transformer-flight-maintenance",
                daemon=True,
            )
            self._thread.start()
        return self

    def shutdown(self, timeout: float | None = None) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._lifecycle_lock:
            thread = self._thread
            self._stop.set()
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                self.logger.event(
                    "flight.maintenance.drain_exceeded",
                    timeoutSeconds=timeout,
                )
                # The runtime-directory lock must remain exclusively owned until
                # all maintenance mutation has stopped.  Python threads cannot
                # be safely killed, so a slow local filesystem/PostgreSQL pass is
                # allowed to finish before application shutdown can release
                # that lock.
                thread.join()

    def run_once(self, *, now: float | None = None) -> MaintenanceResult:
        timestamp = time.time() if now is None else now
        if (
            isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
        ):
            raise ValueError("now must be a finite number")
        timestamp = float(timestamp)
        cutoff = timestamp - self.config.retention_seconds
        if self._queue_reconciler is not None:
            self._queue_reconciler()
        usage = _optional_disk_usage(self.spool)
        if usage is not None:
            self.metrics.set("diskTotalBytes", usage.total)
            self.metrics.set("diskUsedBytes", usage.used)
            self.metrics.set("diskFreeBytes", usage.free)
        recovery_usage = None
        if self.recovery_store is not None:
            recovery_usage = self.recovery_store.disk_usage()
            self.metrics.set(
                "recoveryDiskTotalBytes",
                recovery_usage.total,
            )
            self.metrics.set(
                "recoveryDiskUsedBytes",
                recovery_usage.used,
            )
            self.metrics.set(
                "recoveryDiskFreeBytes",
                recovery_usage.free,
            )

        expired_tickets = self.ledger.delete_expired_tickets(now=timestamp)
        expired_input_jobs = tuple(self.ledger.expire_input_waits(
            timeout_seconds=self.config.input_idle_timeout_seconds,
            now=timestamp,
        ))
        deleted_jobs = tuple(self.ledger.delete_terminal_jobs_before(cutoff))
        with self._pending_lock:
            self._pending_job_directories.update(deleted_jobs)
            pending = tuple(sorted(self._pending_job_directories))

        removed: list[str] = []
        missing: list[str] = []
        failed: list[str] = []
        for job_id in pending:
            directory = self.spool.job_directory(job_id)
            try:
                existed = self.spool.remove(directory)
            except OSError as exc:
                failed.append(job_id)
                self.logger.event(
                    "flight.maintenance.job_directory_failed",
                    jobId=job_id,
                    errorType=type(exc).__name__,
                )
                continue
            with self._pending_lock:
                self._pending_job_directories.discard(job_id)
            (removed if existed else missing).append(job_id)

        recovery_removed = 0
        recovery_failed = 0
        if self.recovery_store is not None:
            recovery_job_ids = set(deleted_jobs)
            recovery_job_ids.update(
                self.ledger.terminal_recovery_job_ids()
            )
            for job_id in sorted(recovery_job_ids):
                directory = self.recovery_store.job_directory(job_id)
                try:
                    if self.recovery_store.remove(directory):
                        recovery_removed += 1
                except OSError as exc:
                    recovery_failed += 1
                    self.logger.event(
                        "flight.maintenance.recovery_directory_failed",
                        jobId=job_id,
                        errorType=type(exc).__name__,
                    )

        completed_models: list[str] = []
        removed_models: list[str] = []
        missing_models: list[str] = []
        failed_models: list[str] = []
        if self.model_deletions is not None:
            for model_ref in self.model_deletions.pending_deletions():
                directory = self.spool.model_directory(model_ref)
                try:
                    existed = self.spool.remove(directory)
                except OSError as exc:
                    failed_models.append(model_ref)
                    self.logger.event(
                        "flight.maintenance.model_directory_failed",
                        modelRef=model_ref,
                        errorType=type(exc).__name__,
                    )
                    continue
                if self.model_deletions.complete_deletion(model_ref):
                    completed_models.append(model_ref)
                (removed_models if existed else missing_models).append(
                    model_ref
                )

        with self._pending_lock:
            still_pending = tuple(sorted(self._pending_job_directories))
        result = MaintenanceResult(
            now=timestamp,
            cutoff=cutoff,
            expired_tickets=expired_tickets,
            expired_input_jobs=expired_input_jobs,
            deleted_jobs=deleted_jobs,
            removed_job_directories=tuple(removed),
            missing_job_directories=tuple(missing),
            failed_job_directories=tuple(failed),
            pending_job_directories=still_pending,
            completed_model_deletions=tuple(completed_models),
            removed_model_directories=tuple(removed_models),
            missing_model_directories=tuple(missing_models),
            failed_model_directories=tuple(failed_models),
        )
        log_fields = {
            "expiredTickets": expired_tickets,
            "expiredInputJobs": len(expired_input_jobs),
            "deletedJobs": len(deleted_jobs),
            "removedJobDirectories": len(removed),
            "missingJobDirectories": len(missing),
            "failedJobDirectories": len(failed),
            "removedRecoveryDirectories": recovery_removed,
            "failedRecoveryDirectories": recovery_failed,
            "completedModelDeletions": len(completed_models),
            "removedModelDirectories": len(removed_models),
            "missingModelDirectories": len(missing_models),
            "failedModelDirectories": len(failed_models),
        }
        if usage is not None:
            log_fields.update(
                diskTotalBytes=usage.total,
                diskUsedBytes=usage.used,
                diskFreeBytes=usage.free,
            )
        if recovery_usage is not None:
            log_fields.update(
                recoveryDiskTotalBytes=recovery_usage.total,
                recoveryDiskUsedBytes=recovery_usage.used,
                recoveryDiskFreeBytes=recovery_usage.free,
            )
        self.logger.event(
            "flight.maintenance.completed",
            **log_fields,
        )
        self.metrics.add("maintenanceRuns")
        self.metrics.add("expiredTickets", expired_tickets)
        self.metrics.add("inputTimeouts", len(expired_input_jobs))
        self.metrics.add("retainedJobsDeleted", len(deleted_jobs))
        self.metrics.add("modelsDeleted", len(completed_models))
        return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self.logger.event(
                    "flight.maintenance.failed",
                    errorType=type(exc).__name__,
                )
            self._stop.wait(self.interval_seconds)


def _optional_disk_usage(store: object) -> DiskUsage | None:
    reader = cast(object, getattr(store, "disk_usage", None))
    if reader is None:
        return None
    if not callable(reader):
        raise TypeError("artifact disk_usage must be callable")
    return cast(Callable[[], DiskUsage], reader)()


__all__ = ["MaintenanceResult", "MaintenanceService"]
