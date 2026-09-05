from __future__ import annotations

import threading
from dataclasses import replace

from app.contracts.metrics.fit_run.v5 import RUN_INDEX
from app.contracts.metrics.v5 import POINT_INDEX
from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.artifacts.telemetry.projection import (
    TrainingMetricsProjection,
)
from app.service.application.ports.telemetry import (
    RetryableMetricsDeliveryError,
)
from app.service.application.telemetry.publisher import (
    MetricsPublisher,
    _retry_delay,
)
from app.service.application.telemetry.records import (
    FitRunSummaryArtifactRecord,
    MetricsOutboxRecord,
    TelemetryArtifactCleanup,
    TrainingMetricsArtifactRecord,
)
from tests.support.telemetry import (
    create_test_metrics_artifact,
    create_test_run_summary_artifact,
)


def _entry() -> MetricsOutboxRecord:
    artifact = TrainingMetricsArtifactRecord(
        model_ref="mdl_" + "1" * 32,
        format="transformer.training-metrics.v5",
        media_type="application/x-ndjson",
        relative_path=(
            "11111111-1111-4111-8111-111111111111/metrics.jsonl"
        ),
        byte_count=100,
        sha256="2" * 64,
        row_count=1,
        job_id="11111111-1111-4111-8111-111111111111",
        attempt_id="22222222-2222-4222-8222-222222222222",
        attempt=1,
        application_version="0.1.10",
        git_commit="3" * 40,
        created_at=1.0,
    )
    return MetricsOutboxRecord(
        training_metrics=artifact,
        projection_version="transformer.metrics.v5",
        status="PENDING",
        cursor=0,
        attempts=0,
        next_attempt_at=1.0,
        created_at=1.0,
    )


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class _Projection:
    def __init__(self, point_count: int) -> None:
        self._points = tuple(
            {"eventId": f"{index:064x}"}
            for index in range(point_count)
        )

    def points(self, _entry, *, deployment_id):
        assert deployment_id == "hp800g9.home"
        return self._points

    def run_summary_document(self, _entry, *, deployment_id):
        assert deployment_id == "hp800g9.home"
        return {"summaryId": "e" * 64}


class _Outbox:
    def __init__(self, entry: MetricsOutboxRecord) -> None:
        self.entry = entry
        self.delivered = threading.Event()
        self.retried = threading.Event()
        self.retry_delay: float | None = None
        self.maintenance_runs = 0

    def next_pending(self):
        return self.entry if self.entry.status == "PENDING" else None

    def advance(self, job_id, *, expected_cursor, cursor):
        assert job_id == self.entry.training_metrics.job_id
        if self.entry.cursor != expected_cursor:
            return False
        self.entry = replace(self.entry, cursor=cursor)
        return True

    def retry(
        self,
        job_id,
        *,
        expected_cursor,
        delay_seconds,
        error_code,
        error_message,
    ):
        assert job_id == self.entry.training_metrics.job_id
        assert self.entry.cursor == expected_cursor
        assert error_code == "DELIVERY_RETRYABLE"
        assert error_message
        self.retry_delay = delay_seconds
        self.entry = replace(self.entry, status="RETRY_WAIT")
        self.retried.set()
        return True

    def block(self, *_args, **_kwargs):
        raise AssertionError("delivery must not be blocked")

    def discard(
        self,
        job_id,
        *,
        expected_cursor,
        error_code,
        error_message,
    ):
        assert job_id == self.entry.training_metrics.job_id
        assert self.entry.cursor == expected_cursor
        assert error_code == "DELIVERY_EXPIRED"
        assert error_message
        self.entry = replace(self.entry, status="CANCELLED")
        self.delivered.set()
        return True

    def complete(self, job_id, *, expected_cursor):
        assert job_id == self.entry.training_metrics.job_id
        if self.entry.cursor != expected_cursor:
            return False
        self.entry = replace(self.entry, status="DELIVERED")
        self.delivered.set()
        return True

    def purge_terminal(self, *, older_than_seconds):
        assert older_than_seconds > 0
        return ()

    def retained_run_ids(self):
        return {self.entry.training_metrics.job_id}

    def backlog(self):
        self.maintenance_runs += 1
        return (0, 0, None)


class _Sink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def create_documents(self, index, documents, *, id_field):
        self.calls.append((index, len(documents), id_field))


class _Storage:
    def __init__(self) -> None:
        self.removed: list[tuple[str, ...]] = []

    def remove_telemetry_artifacts(self, relative_paths):
        self.removed.append(tuple(relative_paths))


def test_publisher_delivers_bounded_point_chunks_before_run_summary():
    outbox = _Outbox(_entry())
    sink = _Sink()
    publisher = MetricsPublisher(
        outbox,
        _Projection(501),
        sink,
        _Storage(),
        deployment_id="hp800g9.home",
        logger=_Logger(),
        metrics=OperationalMetrics(),
    ).start()
    try:
        assert outbox.delivered.wait(2.0)
    finally:
        publisher.shutdown(2.0)

    assert sink.calls == [
        (POINT_INDEX, 500, "eventId"),
        (POINT_INDEX, 1, "eventId"),
        (RUN_INDEX, 1, "summaryId"),
    ]
    assert outbox.maintenance_runs >= 1


def test_terminal_retention_removes_only_the_run_artifacts():
    entry = replace(_entry(), status="DELIVERED")

    class RetentionOutbox(_Outbox):
        def __init__(self, retained: MetricsOutboxRecord) -> None:
            super().__init__(retained)
            self.purged = threading.Event()

        def purge_terminal(self, *, older_than_seconds):
            assert older_than_seconds > 0
            if self.purged.is_set():
                return ()
            self.purged.set()
            return (
                TelemetryArtifactCleanup(
                    job_id=self.entry.training_metrics.job_id,
                    relative_paths=(
                        self.entry.training_metrics.relative_path,
                        f"{self.entry.training_metrics.job_id}/run-summary.json",
                    ),
                ),
            )

    outbox = RetentionOutbox(entry)
    storage = _Storage()
    publisher = MetricsPublisher(
        outbox,
        _Projection(1),
        _Sink(),
        storage,
        deployment_id="hp800g9.home",
        logger=_Logger(),
        metrics=OperationalMetrics(),
    ).start()
    try:
        assert outbox.purged.wait(2.0)
    finally:
        publisher.shutdown(2.0)

    assert storage.removed == [(
        entry.training_metrics.relative_path,
        f"{entry.training_metrics.job_id}/run-summary.json",
    )]


def test_retryable_delivery_keeps_the_outbox_pending_for_later_replay():
    class RetrySink:
        def create_documents(self, _index, _documents, *, id_field):
            assert id_field == "eventId"
            raise RetryableMetricsDeliveryError("temporarily unavailable")

    outbox = _Outbox(_entry())
    publisher = MetricsPublisher(
        outbox,
        _Projection(1),
        RetrySink(),
        _Storage(),
        deployment_id="hp800g9.home",
        logger=_Logger(),
        metrics=OperationalMetrics(),
        random_value=lambda: 0.5,
        wall_clock=lambda: 2.0,
    ).start()
    try:
        assert outbox.retried.wait(2.0)
    finally:
        publisher.shutdown(2.0)

    assert outbox.retry_delay == 1.0
    assert outbox.entry.cursor == 0


def test_retryable_delivery_is_discarded_after_the_retry_budget():
    class RetrySink:
        def create_documents(self, _index, _documents, *, id_field):
            assert id_field == "eventId"
            raise RetryableMetricsDeliveryError("still unavailable")

    outbox = _Outbox(replace(_entry(), attempts=287))
    publisher = MetricsPublisher(
        outbox,
        _Projection(1),
        RetrySink(),
        _Storage(),
        deployment_id="hp800g9.home",
        logger=_Logger(),
        metrics=OperationalMetrics(),
        wall_clock=lambda: 2.0,
    ).start()
    try:
        assert outbox.delivered.wait(2.0)
    finally:
        publisher.shutdown(2.0)

    assert outbox.entry.status == "CANCELLED"
    assert outbox.retry_delay is None


def test_unexpected_delivery_failure_is_blocked_instead_of_retried_forever():
    class BrokenProjection(_Projection):
        def points(self, _entry, *, deployment_id):
            assert deployment_id == "hp800g9.home"
            raise RuntimeError("unexpected projection failure")

    class BlockingOutbox(_Outbox):
        def block(
            self,
            job_id,
            *,
            expected_cursor,
            error_code,
            error_message,
        ):
            assert job_id == self.entry.training_metrics.job_id
            assert expected_cursor == self.entry.cursor
            assert error_code == "DELIVERY_INTERNAL"
            assert error_message
            self.entry = replace(self.entry, status="BLOCKED")
            self.delivered.set()
            return True

    outbox = BlockingOutbox(_entry())
    publisher = MetricsPublisher(
        outbox,
        BrokenProjection(1),
        _Sink(),
        _Storage(),
        deployment_id="hp800g9.home",
        logger=_Logger(),
        metrics=OperationalMetrics(),
    ).start()
    try:
        assert outbox.delivered.wait(2.0)
    finally:
        publisher.shutdown(2.0)

    assert outbox.entry.status == "BLOCKED"


def test_shutdown_timeout_does_not_fail_the_service():
    class BlockingSink:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def create_documents(self, _index, _documents, *, id_field):
            assert id_field == "eventId"
            self.entered.set()
            self.release.wait(2.0)

    sink = BlockingSink()
    logger = _Logger()
    publisher = MetricsPublisher(
        _Outbox(_entry()),
        _Projection(1),
        sink,
        _Storage(),
        deployment_id="hp800g9.home",
        logger=logger,
        metrics=OperationalMetrics(),
    ).start()
    assert sink.entered.wait(2.0)

    publisher.shutdown(0.0)
    sink.release.set()
    publisher.shutdown(2.0)

    assert any(
        event == "metrics.publisher.drain_exceeded"
        for event, _fields in logger.events
    )


def test_current_projection_verifies_immutable_run_summary(tmp_path):
    spool = Spool(
        str(tmp_path / "runtime"),
        str(tmp_path / "models"),
    ).initialize()
    entry = _entry()
    artifact = create_test_metrics_artifact(
        spool,
        model_ref=entry.training_metrics.model_ref,
        job_id=entry.training_metrics.job_id,
        attempt_id=entry.training_metrics.attempt_id,
        attempt=entry.training_metrics.attempt,
    )
    summary = create_test_run_summary_artifact(
        spool,
        model_ref=entry.training_metrics.model_ref,
        job_id=entry.training_metrics.job_id,
        attempt_id=entry.training_metrics.attempt_id,
        attempt=entry.training_metrics.attempt,
    )
    current = replace(
        entry,
        training_metrics=replace(
            entry.training_metrics,
            format="transformer.training-metrics.v5",
            relative_path=artifact.relative_path,
            byte_count=artifact.byte_count,
            sha256=artifact.sha256,
            row_count=artifact.row_count,
            git_commit="0" * 40,
        ),
        run_summary=FitRunSummaryArtifactRecord(
            model_ref=entry.training_metrics.model_ref,
            format="transformer.fit-run-summary.v5",
            media_type="application/json",
            relative_path=summary.relative_path,
            byte_count=summary.byte_count,
            sha256=summary.sha256,
            job_id=entry.training_metrics.job_id,
            attempt_id=entry.training_metrics.attempt_id,
            attempt=entry.training_metrics.attempt,
            application_version=entry.training_metrics.application_version,
            git_commit="0" * 40,
            created_at=10.0,
        ),
        projection_version="transformer.metrics.v5",
    )

    projection = TrainingMetricsProjection(spool)
    points = projection.points(current, deployment_id="hp800g9.home")
    document = projection.run_summary_document(
        current,
        deployment_id="hp800g9.home",
    )

    assert points
    assert document["runId"] == entry.training_metrics.job_id
    assert document["modelRef"] == entry.training_metrics.model_ref


def test_retry_delay_remains_inside_the_operational_bounds():
    assert _retry_delay(0, 0.0) == 1.0
    assert _retry_delay(100, 1.0) == 300.0
