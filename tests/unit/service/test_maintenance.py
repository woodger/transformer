from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.postgres.ledger.maintenance import (
    MaintenanceLedgerSlice,
)
from app.service.bootstrap.maintenance import MaintenanceService
from app.service.domain.job import ErrorCode, ExecutionState, InputState


class RecordingLogger:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))


def test_startup_reconciliation_terminalizes_every_unfinished_job():
    waiting = _job(
        "waiting",
        ExecutionState.WAITING_INPUT,
        InputState.OPEN,
    )
    queued = _job("queued", ExecutionState.QUEUED, InputState.CLOSED)
    running = _job("running", ExecutionState.RUNNING, InputState.OPEN)
    retrying = _job("retrying", ExecutionState.RETRYING, InputState.CLOSED)
    cancelling = _job(
        "cancelling",
        ExecutionState.CANCELLING,
        InputState.OPEN,
    )
    running_attempt = _attempt(ExecutionState.RUNNING)
    cancelling_attempt = _attempt(ExecutionState.RUNNING)
    attempts = {
        (running.job_id, running.attempt): running_attempt,
        (cancelling.job_id, cancelling.attempt): cancelling_attempt,
    }
    session = SimpleNamespace(
        execute=lambda _statement: SimpleNamespace(all=lambda: []),
        scalars=lambda _statement: (
            waiting,
            queued,
            running,
            retrying,
            cancelling,
        ),
        get=lambda _model, identity, **_kwargs: attempts.get(identity),
    )
    ledger = MaintenanceLedgerSlice(
        SimpleNamespace(
            database=SimpleNamespace(
                transaction=lambda: _transaction(session),
            )
        )
    )

    result = ledger.reconcile_startup_jobs(now=100.0)

    assert waiting.execution_state == ExecutionState.FAILED.value
    assert queued.execution_state == ExecutionState.FAILED.value
    assert running.execution_state == ExecutionState.FAILED.value
    assert retrying.execution_state == ExecutionState.FAILED.value
    assert cancelling.execution_state == ExecutionState.CANCELLED.value
    assert waiting.input_state == InputState.ABORTED.value
    assert running.input_state == InputState.ABORTED.value
    assert cancelling.input_state == InputState.ABORTED.value
    assert queued.input_state == InputState.CLOSED.value
    assert retrying.input_state == InputState.CLOSED.value
    assert running_attempt.status == ExecutionState.FAILED.value
    assert running_attempt.error_code == ErrorCode.EXECUTION_INTERRUPTED.value
    assert cancelling_attempt.status == ExecutionState.CANCELLED.value
    assert result["failed_waiting_input_jobs"] == [waiting.job_id]
    assert result["failed_queued_jobs"] == [queued.job_id]
    assert result["failed_running_jobs"] == [running.job_id]
    assert result["failed_retrying_jobs"] == [retrying.job_id]
    assert result["cancelled_jobs"] == [cancelling.job_id]


@contextmanager
def _transaction(session):
    yield session


def _job(job_id, execution_state, input_state):
    return SimpleNamespace(
        job_id=job_id,
        attempt=1,
        execution_state=execution_state.value,
        input_state=input_state.value,
        waiting_for_input=False,
        waiting_input_ordinal=None,
        input_waiting_since=None,
        acquire_grace_until=None,
        error_code=None,
        error_message=None,
        revision=1,
        finished_at=None,
        updated_at=None,
    )


def _attempt(status):
    return SimpleNamespace(
        status=status.value,
        error_code=None,
        error_message=None,
        finished_at=None,
    )


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


def test_model_row_is_purged_after_directory_removal():
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


def test_missing_model_directory_still_purges_pending_model_row():
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
    drain_exceeded = threading.Event()
    shutdown_finished = threading.Event()
    completed = []
    failures = []

    def log(event, **_fields):
        if event == "flight.maintenance.drain_exceeded":
            drain_exceeded.set()

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            entered.set()
            assert release.wait(5.0)
            completed.append("mutation")
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
        logger=SimpleNamespace(event=log),
        metrics=OperationalMetrics(),
    ).start()

    def shutdown():
        try:
            maintenance.shutdown(timeout=0.001)
            completed.append("shutdown")
        except BaseException as exc:
            failures.append(exc)
        finally:
            shutdown_finished.set()

    shutdown_thread = threading.Thread(target=shutdown)
    try:
        assert entered.wait(1.0)
        shutdown_thread.start()
        assert drain_exceeded.wait(1.0)
        assert not shutdown_finished.wait(0.05)
        release.set()
        assert shutdown_finished.wait(1.0)
    finally:
        release.set()
        if shutdown_thread.ident is not None:
            shutdown_thread.join(timeout=2.0)
        maintenance.shutdown(timeout=2.0)

    assert not shutdown_thread.is_alive()
    assert failures == []
    assert completed == ["mutation", "shutdown"]
    assert maintenance.running is False
