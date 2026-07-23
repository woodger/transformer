from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

from app.flight.observability import JsonLogger, OperationalMetrics


@dataclass(frozen=True)
class MaintenanceResult:
    now: float
    cutoff: float
    expired_tickets: int
    deleted_jobs: tuple[str, ...]
    removed_job_directories: tuple[str, ...]
    missing_job_directories: tuple[str, ...]
    failed_job_directories: tuple[str, ...]
    pending_job_directories: tuple[str, ...]


class MaintenanceService:
    """Periodically expire tickets and retained terminal jobs.

    Terminal ledger rows are deleted before their server-owned job directories.
    Reversing that order could leave live ledger references to missing artifacts.
    A process crash between the two operations can therefore leave an orphan job
    directory. Startup reconciliation must remove ``spool/jobs/<jobId>`` entries
    whose canonical job IDs are absent from the ledger-provided known-job set.
    The in-memory pending set only closes retry gaps while this process remains
    alive; it is deliberately not treated as durable state.

    Model directories are never considered here. The ledger retention primitive
    excludes jobs that produced a model generation, preserving both the model
    row and its producing job's immutable provenance.
    """

    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        interval_seconds: float = 60.0,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
    ):
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a positive finite number")
        if config.retention_seconds <= 0:
            raise ValueError("retention_seconds must be greater than zero")

        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.interval_seconds = float(interval_seconds)
        self.logger = logger or JsonLogger()
        self.metrics = metrics or OperationalMetrics()

        self._lifecycle_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_job_directories: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> "MaintenanceService":
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
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
        ):
            raise ValueError("now must be a finite number")
        timestamp = float(timestamp)
        cutoff = timestamp - self.config.retention_seconds
        usage = None
        watermark = getattr(self.config, "disk_min_free_bytes", None)
        disk_usage = getattr(self.spool, "disk_usage", None)
        if disk_usage is not None and watermark is not None:
            usage = disk_usage()
            watermark_exceeded = usage.free < watermark
            self.metrics.set("diskTotalBytes", usage.total)
            self.metrics.set("diskUsedBytes", usage.used)
            self.metrics.set("diskFreeBytes", usage.free)
            self.metrics.set("diskWatermarkBytes", watermark)
            self.metrics.set("diskWatermarkExceeded", watermark_exceeded)

        expired_tickets = self.ledger.delete_expired_tickets(now=timestamp)
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

        with self._pending_lock:
            still_pending = tuple(sorted(self._pending_job_directories))
        result = MaintenanceResult(
            now=timestamp,
            cutoff=cutoff,
            expired_tickets=expired_tickets,
            deleted_jobs=deleted_jobs,
            removed_job_directories=tuple(removed),
            missing_job_directories=tuple(missing),
            failed_job_directories=tuple(failed),
            pending_job_directories=still_pending,
        )
        log_fields = {
            "expiredTickets": expired_tickets,
            "deletedJobs": len(deleted_jobs),
            "removedJobDirectories": len(removed),
            "missingJobDirectories": len(missing),
            "failedJobDirectories": len(failed),
        }
        if usage is not None:
            log_fields.update(
                diskTotalBytes=usage.total,
                diskUsedBytes=usage.used,
                diskFreeBytes=usage.free,
                diskWatermarkBytes=watermark,
                diskWatermarkExceeded=watermark_exceeded,
            )
        self.logger.event(
            "flight.maintenance.completed",
            **log_fields,
        )
        self.metrics.add("maintenanceRuns")
        self.metrics.add("expiredTickets", expired_tickets)
        self.metrics.add("retainedJobsDeleted", len(deleted_jobs))
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


__all__ = ["MaintenanceResult", "MaintenanceService"]
