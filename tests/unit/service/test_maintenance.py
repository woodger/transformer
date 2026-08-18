from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.bootstrap.config import FlightServiceConfig
from app.service.bootstrap.maintenance import MaintenanceService
from app.service.domain.job import ErrorCode, ExecutionState
from tests.support.flight_v4_helpers import (
    close_input,
    commit_input,
    create_fit,
    internal_data_contract,
    model_config,
)


class RecordingLogger:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))


def _config(tmp_path, **overrides):
    base = FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        allow_plaintext=True,
        retention_seconds=50,
        input_idle_timeout_seconds=20,
    )
    return replace(base, **overrides).validate()


def _components(tmp_path, ledger, **overrides):
    config = _config(tmp_path, **overrides)
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    maintenance = MaintenanceService(
        config,
        ledger,
        spool,
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    )
    return config, spool, maintenance


def _cancelled_job(ledger, *, finished_at):
    job = create_fit(ledger, now=finished_at - 1)
    return ledger.transition_job(
        job["job_id"],
        ExecutionState.CANCELLED,
        updates={"finished_at": finished_at},
        now=finished_at,
    )


def _job_marker(spool, job_id):
    path = Path(spool.job_directory(job_id)) / "attempts" / "marker.log"
    spool.atomic_write_bytes(str(path), b"server-owned job data")
    return path


def test_retention_removes_heavy_job_but_keeps_identity_tombstone(
    tmp_path,
    postgres_ledger,
):
    _, spool, maintenance = _components(tmp_path, postgres_ledger)
    expired = _cancelled_job(postgres_ledger, finished_at=10.0)
    boundary = _cancelled_job(postgres_ledger, finished_at=50.0)
    marker = _job_marker(spool, expired["job_id"])

    result = maintenance.run_once(now=100.0)

    assert result.cutoff == 50.0
    assert result.deleted_jobs == (expired["job_id"],)
    assert result.removed_job_directories == (expired["job_id"],)
    assert not marker.exists()
    assert postgres_ledger.get_job(expired["job_id"]) is None
    assert postgres_ledger.get_job_identity(expired["job_id"]) is not None
    assert postgres_ledger.get_job(boundary["job_id"]) is not None


def test_input_timeout_starts_only_after_worker_confirms_frontier_wait(
    tmp_path,
    postgres_ledger,
):
    _, _, maintenance = _components(tmp_path, postgres_ledger)
    never_started = create_fit(postgres_ledger, now=1.0)
    running = create_fit(postgres_ledger, now=1.0)
    commit_input(postgres_ledger, running, 0, now=2.0)
    claimed = postgres_ledger.claim_execution_job(
        running["job_id"],
        "cpu",
        now=3.0,
    )
    assert claimed is not None
    commit_input(postgres_ledger, running, 2, now=3.5)
    current = postgres_ledger.get_job(running["job_id"])
    assert current["input_revision"] == 2
    assert current["next_input_ordinal"] == 1
    assert postgres_ledger.mark_input_waiting(
        running["job_id"],
        claimed.attempt,
        attempt_id=claimed.attempt_id,
        next_ordinal=1,
        input_revision=1,
        now=4.0,
    )

    result = maintenance.run_once(now=25.0)

    assert result.expired_input_jobs == (running["job_id"],)
    untouched = postgres_ledger.get_job(never_started["job_id"])
    assert untouched["execution_state"] == ExecutionState.WAITING_INPUT.value
    timed_out = postgres_ledger.get_job(running["job_id"])
    assert timed_out["execution_state"] == ExecutionState.FAILED.value
    assert timed_out["error_code"] == ErrorCode.INPUT_TIMEOUT.value


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


def test_published_model_outlives_its_producing_job(
    tmp_path,
    postgres_ledger,
):
    _, spool, maintenance = _components(tmp_path, postgres_ledger)
    job = create_fit(postgres_ledger, now=1.0)
    commit_input(postgres_ledger, job, 0, now=2.0)
    close_input(postgres_ledger, job, now=3.0)
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu", now=4.0)
    model_ref = f"mdl_{uuid.uuid4().hex}"
    checkpoint = spool.model_checkpoint_path(model_ref)
    metadata_path = spool.model_metadata_path(model_ref)
    spool.atomic_write_bytes(checkpoint, b"checkpoint")
    spool.atomic_write_json(metadata_path, {"modelRef": model_ref})
    raw = Path(checkpoint).read_bytes()
    postgres_ledger.publish_model(
        job["job_id"],
        running.attempt,
        attempt_id=running.attempt_id,
        model_ref=model_ref,
        label="daily",
        generation=None,
        checkpoint_path=spool.model_relative_path(checkpoint),
        metadata_path=spool.model_relative_path(metadata_path),
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        metadata={
            "model_config": model_config().to_dict(),
            "data_contract": internal_data_contract(),
        },
        result={"modelRef": model_ref},
        now=10.0,
    )

    result = maintenance.run_once(now=100.0)

    assert result.deleted_jobs == (job["job_id"],)
    assert postgres_ledger.get_job(job["job_id"]) is None
    model = postgres_ledger.get_model(model_ref)
    assert model["producing_job_id"] is None
    assert Path(checkpoint).is_file()
    assert Path(metadata_path).is_file()
    assert Path(spool.model_metrics_path(model_ref)).is_file()


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
