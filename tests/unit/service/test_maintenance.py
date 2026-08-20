from __future__ import annotations

import threading
import time
import uuid
from types import SimpleNamespace

from app.service.adapters.observability import OperationalMetrics
from app.service.bootstrap.maintenance import MaintenanceService


class RecordingLogger:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))


def test_ledger_row_is_deleted_before_job_directory_is_touched():
    job_id = str(uuid.uuid4())
    events = []

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            events.append(("tickets", now))
            return 0

        def expire_input_waits(self, *, timeout_seconds, now):
            events.append(("input-timeout", timeout_seconds, now))
            return []

        def delete_terminal_jobs_before(self, cutoff):
            events.append(("ledger", cutoff))
            return [job_id]

    class SpoolDouble:
        def job_directory(self, value):
            return f"/managed/jobs/{value}"

        def remove(self, path):
            events.append(("filesystem", path))
            return True

    maintenance = MaintenanceService(
        SimpleNamespace(
            retention_seconds=10,
            input_idle_timeout_seconds=30,
        ),
        LedgerDouble(),
        SpoolDouble(),
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    )

    maintenance.run_once(now=100.0)

    assert events == [
        ("tickets", 100.0),
        ("input-timeout", 30, 100.0),
        ("ledger", 90.0),
        ("filesystem", f"/managed/jobs/{job_id}"),
    ]


def test_model_tombstone_is_finalized_after_directory_removal():
    model_ref = f"mdl_{uuid.uuid4().hex}"
    events = []

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            return 0

        def expire_input_waits(self, **_kwargs):
            return []

        def delete_terminal_jobs_before(self, _cutoff):
            return []

    class SpoolDouble:
        def model_directory(self, value):
            return f"/managed/models/{value}"

        def remove(self, path):
            events.append(("filesystem", path))
            return True

    class ModelDeletionDouble:
        def pending_deletions(self, *, limit=100):
            assert limit == 100
            return (model_ref,)

        def complete_deletion(self, value):
            events.append(("postgres", value))
            return True

    maintenance = MaintenanceService(
        SimpleNamespace(
            retention_seconds=10,
            input_idle_timeout_seconds=30,
        ),
        LedgerDouble(),
        SpoolDouble(),
        model_deletions=ModelDeletionDouble(),
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    )

    result = maintenance.run_once(now=100.0)

    assert events == [
        ("filesystem", f"/managed/models/{model_ref}"),
        ("postgres", model_ref),
    ]
    assert result.completed_model_deletions == (model_ref,)
    assert result.removed_model_directories == (model_ref,)


def test_missing_model_directory_still_finalizes_pending_tombstone():
    model_ref = f"mdl_{uuid.uuid4().hex}"
    completed = []

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            return 0

        def expire_input_waits(self, **_kwargs):
            return []

        def delete_terminal_jobs_before(self, _cutoff):
            return []

    class SpoolDouble:
        def model_directory(self, value):
            return f"/managed/models/{value}"

        def remove(self, _path):
            return False

    class ModelDeletionDouble:
        def pending_deletions(self, *, limit=100):
            return (model_ref,)

        def complete_deletion(self, value):
            completed.append(value)
            return True

    maintenance = MaintenanceService(
        SimpleNamespace(
            retention_seconds=10,
            input_idle_timeout_seconds=30,
        ),
        LedgerDouble(),
        SpoolDouble(),
        model_deletions=ModelDeletionDouble(),
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    )

    result = maintenance.run_once(now=100.0)

    assert completed == [model_ref]
    assert result.missing_model_directories == (model_ref,)


def test_model_directory_failure_leaves_deletion_pending():
    model_ref = f"mdl_{uuid.uuid4().hex}"
    completed = []

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            return 0

        def expire_input_waits(self, **_kwargs):
            return []

        def delete_terminal_jobs_before(self, _cutoff):
            return []

    class SpoolDouble:
        def model_directory(self, value):
            return f"/managed/models/{value}"

        def remove(self, _path):
            raise OSError("injected filesystem failure")

    class ModelDeletionDouble:
        def pending_deletions(self, *, limit=100):
            return (model_ref,)

        def complete_deletion(self, value):
            completed.append(value)
            return True

    maintenance = MaintenanceService(
        SimpleNamespace(
            retention_seconds=10,
            input_idle_timeout_seconds=30,
        ),
        LedgerDouble(),
        SpoolDouble(),
        model_deletions=ModelDeletionDouble(),
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    )

    result = maintenance.run_once(now=100.0)

    assert completed == []
    assert result.failed_model_directories == (model_ref,)


def test_periodic_maintenance_shutdown_waits_for_active_mutation():
    entered = threading.Event()
    release = threading.Event()

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            entered.set()
            release.wait(1.0)
            return 0

        def expire_input_waits(self, **_kwargs):
            return []

        def delete_terminal_jobs_before(self, _cutoff):
            return []

    maintenance = MaintenanceService(
        SimpleNamespace(
            retention_seconds=10,
            input_idle_timeout_seconds=30,
        ),
        LedgerDouble(),
        SimpleNamespace(),
        interval_seconds=60,
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    ).start()
    assert entered.wait(1.0)
    release_timer = threading.Timer(0.05, release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        maintenance.shutdown(timeout=0.001)
    finally:
        release_timer.join(timeout=1.0)

    assert time.monotonic() - started >= 0.02
    assert maintenance.running is False
