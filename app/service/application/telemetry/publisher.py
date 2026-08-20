from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable

from app.contracts.metrics.fit_run.v2 import RUN_INDEX
from app.contracts.metrics.v3 import POINT_INDEX
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.telemetry import (
    BlockedMetricsDeliveryError,
    MetricsArtifactProjection,
    MetricsDocumentSink,
    MetricsOutboxRepository,
    RetryableMetricsDeliveryError,
    TelemetryArtifactStorage,
)
from app.service.application.telemetry.records import MetricsOutboxRecord

_MAX_BULK_DOCUMENTS = 500
_TERMINAL_RETENTION_SECONDS = 7 * 24 * 60 * 60
_BACKLOG_ENTRY_LIMIT = 10_000
_BACKLOG_BYTE_LIMIT = 10 * 1024 * 1024 * 1024
_BACKLOG_AGE_LIMIT_SECONDS = 30 * 24 * 60 * 60
_MAX_DELIVERY_ATTEMPTS = 288
_MAX_DELIVERY_AGE_SECONDS = 24 * 60 * 60


class MetricsPublisher:
    """Deliver immutable metrics projections after the application commit."""

    def __init__(
        self,
        outbox: MetricsOutboxRepository,
        projection: MetricsArtifactProjection,
        sink: MetricsDocumentSink,
        storage: TelemetryArtifactStorage,
        *,
        deployment_id: str,
        logger: EventLogger,
        metrics: OperationalMetricSink,
        random_value: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.outbox = outbox
        self.projection = projection
        self.sink = sink
        self.storage = storage
        self.deployment_id = deployment_id
        self.logger = logger
        self.metrics = metrics
        self._random = random_value
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> MetricsPublisher:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="transformer-metrics-publisher",
            daemon=True,
        )
        self._thread.start()
        return self

    def shutdown(self, timeout: float | None = None) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                self.metrics.add("metricsPublisherDrainExceeded")
                self.logger.event("metrics.publisher.drain_exceeded")

    def _run(self) -> None:
        next_maintenance = 0.0
        while not self._stop.is_set():
            try:
                now = self._monotonic()
                if now >= next_maintenance:
                    self._maintenance()
                    next_maintenance = now + 60.0
                entry = self.outbox.next_pending()
                if entry is None:
                    self._stop.wait(1.0)
                    continue
                self._deliver(entry)
            except Exception as exc:
                self.metrics.add("metricsPublisherInternalErrors")
                self.logger.event(
                    "metrics.publisher.internal_error",
                    errorType=type(exc).__name__,
                )
                self._stop.wait(1.0)

    def _deliver(self, entry: MetricsOutboxRecord) -> None:
        cursor = entry.cursor
        try:
            points = self.projection.points(
                entry,
                deployment_id=self.deployment_id,
            )
            if cursor > len(points):
                raise ValueError("metrics outbox cursor exceeds projection")
            if cursor < len(points):
                chunk = points[cursor:cursor + _MAX_BULK_DOCUMENTS]
                self.sink.create_documents(
                    POINT_INDEX,
                    chunk,
                    id_field="eventId",
                )
                if self.outbox.advance(
                    entry.training_metrics.job_id,
                    expected_cursor=cursor,
                    cursor=cursor + len(chunk),
                ):
                    self.metrics.add("metricsPointsDelivered", len(chunk))
                return
            run_summary = self.projection.run_summary_document(
                entry,
                deployment_id=self.deployment_id,
            )
            self.sink.create_documents(
                RUN_INDEX,
                (run_summary,),
                id_field="summaryId",
            )
            if self.outbox.complete(
                entry.training_metrics.job_id,
                expected_cursor=cursor,
            ):
                self.metrics.add("metricsRunSummariesDelivered")
                self.logger.event(
                    "metrics.run.delivered",
                    modelRef=entry.training_metrics.model_ref,
                    runId=entry.training_metrics.job_id,
                    points=cursor,
                )
        except RetryableMetricsDeliveryError as exc:
            delivery_age = max(0.0, self._wall_clock() - entry.created_at)
            if (
                entry.attempts + 1 >= _MAX_DELIVERY_ATTEMPTS
                or delivery_age >= _MAX_DELIVERY_AGE_SECONDS
            ):
                if self.outbox.discard(
                    entry.training_metrics.job_id,
                    expected_cursor=cursor,
                    error_code="DELIVERY_EXPIRED",
                    error_message="metrics delivery retry budget was exhausted",
                ):
                    self.metrics.add("metricsDeliveryDropped")
                    self.logger.event(
                        "metrics.delivery.dropped",
                        modelRef=entry.training_metrics.model_ref,
                        runId=entry.training_metrics.job_id,
                        attempts=entry.attempts + 1,
                    )
                return
            delay = _retry_delay(entry.attempts, self._random())
            if self.outbox.retry(
                entry.training_metrics.job_id,
                expected_cursor=cursor,
                delay_seconds=delay,
                error_code="DELIVERY_RETRYABLE",
                error_message=str(exc),
            ):
                self.metrics.add("metricsDeliveryRetries")
                self.logger.event(
                    "metrics.delivery.retry_scheduled",
                    modelRef=entry.training_metrics.model_ref,
                    runId=entry.training_metrics.job_id,
                    delaySeconds=delay,
                )
        except (BlockedMetricsDeliveryError, OSError, TypeError, ValueError) as exc:
            if self.outbox.block(
                entry.training_metrics.job_id,
                expected_cursor=cursor,
                error_code="DELIVERY_INTEGRITY",
                error_message=str(exc),
            ):
                self.metrics.add("metricsDeliveryBlocked")
                self.logger.event(
                    "metrics.delivery.blocked",
                    modelRef=entry.training_metrics.model_ref,
                    runId=entry.training_metrics.job_id,
                    errorType=type(exc).__name__,
                )
        except Exception as exc:
            if self.outbox.block(
                entry.training_metrics.job_id,
                expected_cursor=cursor,
                error_code="DELIVERY_INTERNAL",
                error_message=str(exc),
            ):
                self.metrics.add("metricsDeliveryBlocked")
                self.logger.event(
                    "metrics.delivery.blocked",
                    modelRef=entry.training_metrics.model_ref,
                    runId=entry.training_metrics.job_id,
                    errorType=type(exc).__name__,
                )

    def _maintenance(self) -> None:
        cleanups = self.outbox.purge_terminal(
            older_than_seconds=_TERMINAL_RETENTION_SECONDS,
        )
        for cleanup in cleanups:
            try:
                self.storage.remove_telemetry_artifacts(
                    cleanup.relative_paths
                )
            except (OSError, ValueError) as exc:
                self.metrics.add("trainingTelemetryCleanupErrors")
                self.logger.event(
                    "metrics.cleanup.failed",
                    runId=cleanup.job_id,
                    errorType=type(exc).__name__,
                )
        if cleanups:
            self.metrics.add("metricsOutboxTerminalPurged", len(cleanups))
        entries, byte_count, oldest_age = self.outbox.backlog()
        self.metrics.set("metricsOutboxEntries", entries)
        self.metrics.set("metricsOutboxBytes", byte_count)
        self.metrics.set("metricsOutboxOldestAgeSeconds", oldest_age)
        if (
            entries > _BACKLOG_ENTRY_LIMIT
            or byte_count > _BACKLOG_BYTE_LIMIT
            or (
                oldest_age is not None
                and oldest_age > _BACKLOG_AGE_LIMIT_SECONDS
            )
        ):
            self.logger.event(
                "metrics.outbox.limit_exceeded",
                entries=entries,
                bytes=byte_count,
                oldestAgeSeconds=oldest_age,
            )


def _retry_delay(attempts: int, random_value: float) -> float:
    base = min(300.0, 2.0 ** min(attempts, 8))
    jitter = 0.8 + min(1.0, max(0.0, random_value)) * 0.4
    return min(300.0, max(1.0, base * jitter))


__all__ = ["MetricsPublisher"]
