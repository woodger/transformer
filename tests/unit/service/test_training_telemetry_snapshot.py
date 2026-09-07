from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
import rfc8785

from app.contracts.json_types import JsonValue
from app.service.application.services.training_telemetry_snapshot import (
    TelemetrySnapshotOperation,
    TrainingTelemetrySnapshotCapacityExhausted,
    TrainingTelemetrySnapshotStore,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _store(
    *,
    max_count: int = 2,
    max_total_bytes: int = 1_024,
    max_snapshot_bytes: int = 1_024,
) -> TrainingTelemetrySnapshotStore:
    return TrainingTelemetrySnapshotStore(
        max_count=max_count,
        max_total_bytes=max_total_bytes,
        max_snapshot_bytes=max_snapshot_bytes,
        boot_identity="a" * 32,
    )


def _admit(
    store: TrainingTelemetrySnapshotStore,
    key: str,
    projection: JsonValue,
    *,
    operation: TelemetrySnapshotOperation = "report",
    expires_at: datetime = NOW + timedelta(seconds=900),
) -> object:
    return store.admit(
        (key,),
        {"key": key},
        projection,
        operation=operation,
        expires_at=expires_at,
        now=NOW,
    )


def test_snapshot_count_admission_does_not_evict_live_entry() -> None:
    store = _store(max_count=1)
    retained = _admit(store, "report", {"value": 1})

    with pytest.raises(TrainingTelemetrySnapshotCapacityExhausted) as error:
        _admit(
            store,
            "gradient",
            {"value": 2},
            operation="gradientInteractions",
        )

    assert error.value.operation == "gradientInteractions"
    assert store.get(("report",), now=NOW) is retained
    assert store.usage(now=NOW).count == 1


def test_snapshot_total_byte_admission_is_deterministic_jcs_length() -> None:
    first: JsonValue = {"value": "данные"}
    first_bytes = len(rfc8785.dumps(first))
    store = _store(
        max_total_bytes=first_bytes,
        max_snapshot_bytes=first_bytes,
    )
    _admit(store, "first", first)

    with pytest.raises(TrainingTelemetrySnapshotCapacityExhausted):
        _admit(store, "second", {"value": "second"})

    assert store.usage(now=NOW).byte_count == first_bytes


def test_oversized_single_snapshot_is_rejected() -> None:
    projection: JsonValue = {"value": "oversized"}
    byte_count = len(rfc8785.dumps(projection))
    store = _store(
        max_total_bytes=byte_count,
        max_snapshot_bytes=byte_count - 1,
    )

    with pytest.raises(TrainingTelemetrySnapshotCapacityExhausted):
        _admit(store, "oversized", projection)

    assert store.usage(now=NOW).count == 0


def test_expired_snapshot_releases_capacity() -> None:
    store = _store(max_count=1)
    _admit(store, "first", {"value": 1}, expires_at=NOW + timedelta(seconds=1))

    replacement = store.admit(
        ("second",),
        {"key": "second"},
        {"value": 2},
        operation="report",
        expires_at=NOW + timedelta(seconds=901),
        now=NOW + timedelta(seconds=1),
    )

    assert store.get(("first",), now=NOW + timedelta(seconds=1)) is None
    assert store.get(("second",), now=NOW + timedelta(seconds=1)) is replacement


def test_identical_snapshot_reuse_does_not_consume_capacity_twice() -> None:
    store = _store(max_count=1)
    first_value = {"generation": 1}
    first = store.admit(
        ("same",),
        first_value,
        {"value": 1},
        operation="report",
        expires_at=NOW + timedelta(seconds=100),
        now=NOW,
    )
    usage = store.usage(now=NOW)

    reused = store.admit(
        ("same",),
        {"generation": 2},
        {"value": 1},
        operation="report",
        expires_at=NOW + timedelta(seconds=900),
        now=NOW,
    )

    assert reused is first
    assert store.usage(now=NOW) == usage
    assert store.get(("same",), now=NOW + timedelta(seconds=100)) is first


def test_concurrent_admission_respects_shared_capacity_atomically() -> None:
    store = _store(max_count=1)
    barrier = Barrier(2)

    def admit(key: str) -> str:
        barrier.wait()
        try:
            _admit(store, key, {"value": key})
        except TrainingTelemetrySnapshotCapacityExhausted:
            return "rejected"
        return "admitted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(admit, ("one", "two")))

    assert sorted(outcomes) == ["admitted", "rejected"]
    assert store.usage(now=NOW).count == 1
