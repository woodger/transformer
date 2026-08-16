from __future__ import annotations

import threading
from dataclasses import replace

from app.contracts.metrics.v1 import ARTIFACT_INDEX, POINT_INDEX
from app.service.adapters.observability import OperationalMetrics
from app.service.application.ports.metrics import (
    RetryableMetricsDeliveryError,
)
from app.service.application.services.metrics_publisher import (
    MetricsPublisher,
    _retry_delay,
)
from app.service.domain.records import (
    MetricsOutboxRecord,
    ModelMetricsArtifactRecord,
)


def _entry() -> MetricsOutboxRecord:
    artifact = ModelMetricsArtifactRecord(
        model_ref="mdl_" + "1" * 32,
        format="transformer.training-metrics.v1",
        media_type="application/x-ndjson",
        relative_path="mdl/metrics.jsonl",
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
        artifact=artifact,
        projection_version="inventory.metrics.v1",
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

    def artifact_document(self, _entry, *, deployment_id):
        assert deployment_id == "hp800g9.home"
        return {"artifactId": "f" * 64}


class _Outbox:
    def __init__(self, entry: MetricsOutboxRecord) -> None:
        self.entry = entry
        self.delivered = threading.Event()
        self.retried = threading.Event()
        self.retry_delay: float | None = None
        self.maintenance_runs = 0

    def next_pending(self):
        return self.entry if self.entry.status == "PENDING" else None

    def advance(self, model_ref, *, expected_cursor, cursor):
        if self.entry.cursor != expected_cursor:
            return False
        self.entry = replace(self.entry, cursor=cursor)
        return True

    def retry(
        self,
        model_ref,
        *,
        expected_cursor,
        delay_seconds,
        error_code,
        error_message,
    ):
        assert self.entry.cursor == expected_cursor
        assert error_code == "DELIVERY_RETRYABLE"
        assert error_message
        self.retry_delay = delay_seconds
        self.entry = replace(self.entry, status="RETRY_WAIT")
        self.retried.set()
        return True

    def block(self, *_args, **_kwargs):
        raise AssertionError("delivery must not be blocked")

    def complete(self, model_ref, *, expected_cursor):
        if self.entry.cursor != expected_cursor:
            return False
        self.entry = replace(self.entry, status="DELIVERED")
        self.delivered.set()
        return True

    def purge_delivered(self, *, older_than_seconds):
        assert older_than_seconds > 0
        return 0

    def backlog(self):
        self.maintenance_runs += 1
        return (0, 0, None)


class _Sink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def create_documents(self, index, documents, *, id_field):
        self.calls.append((index, len(documents), id_field))


def test_publisher_delivers_bounded_point_chunks_before_artifact_metadata():
    outbox = _Outbox(_entry())
    sink = _Sink()
    publisher = MetricsPublisher(
        outbox,
        _Projection(501),
        sink,
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
        (ARTIFACT_INDEX, 1, "artifactId"),
    ]
    assert outbox.maintenance_runs >= 1


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
        deployment_id="hp800g9.home",
        logger=_Logger(),
        metrics=OperationalMetrics(),
        random_value=lambda: 0.5,
    ).start()
    try:
        assert outbox.retried.wait(2.0)
    finally:
        publisher.shutdown(2.0)

    assert outbox.retry_delay == 1.0
    assert outbox.entry.cursor == 0


def test_retry_delay_remains_inside_the_operational_bounds():
    assert _retry_delay(0, 0.0) == 1.0
    assert _retry_delay(100, 1.0) == 300.0
