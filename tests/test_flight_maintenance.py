import hashlib
import os
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.database.models import OutputTicket
from app.flight.config import FlightServiceConfig
from app.flight.constants import JobState
from app.flight.maintenance import MaintenanceService
from app.flight.spool import Spool

DIGEST = "a" * 64
_POSTGRES_LEDGER = None


@pytest.fixture(autouse=True)
def _use_postgres_ledger(postgres_ledger):
    global _POSTGRES_LEDGER
    _POSTGRES_LEDGER = postgres_ledger
    try:
        yield
    finally:
        _POSTGRES_LEDGER = None


class RecordingLogger:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))


def service_config(tmp_path, **overrides):
    base = FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        allow_plaintext=True,
        disk_min_free_bytes=1,
        retention_seconds=50,
    )
    return replace(base, **overrides).validate()


def components(tmp_path, **config_overrides):
    config = service_config(tmp_path, **config_overrides)
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    ledger = _POSTGRES_LEDGER
    maintenance = MaintenanceService(
        config,
        ledger,
        spool,
        logger=RecordingLogger(),
    )
    return config, spool, ledger, maintenance


def create_job(ledger, *, operation="fit", now=1.0):
    arguments = {
        "job_id": str(uuid.uuid4()),
        "owner_subject": "inventory",
        "operation": operation,
        "requested_device": "cpu",
        "prediction_column": "out",
        "config_hash": DIGEST,
        "now": now,
    }
    if operation == "fit":
        arguments["model_label"] = "daily-model"
    else:
        arguments["input_model_ref"] = "mdl_existing"
    return ledger.create_job(**arguments)


def cancel_job(ledger, *, finished_at):
    job = create_job(ledger, now=finished_at - 1)
    return ledger.transition_job(
        job["job_id"],
        JobState.CANCELLED,
        updates={"finished_at": finished_at},
        now=finished_at,
    )


def create_job_marker(spool, job_id):
    path = os.path.join(spool.job_directory(job_id), "attempts", "marker.log")
    spool.atomic_write_bytes(path, b"server-owned job data")
    return path


def seal_queue_claim(ledger, job_id):
    ledger.seal_job(
        job_id,
        manifest_hash=DIGEST,
        manifest=[],
        source_width=4,
        feature_dim=2,
        now=2.0,
    )
    ledger.queue_job(job_id, selected_device="cpu", now=3.0)
    return ledger.claim_next_job("cpu", worker_id="maintenance-test", now=4.0)


def test_run_once_uses_retention_cutoff_and_removes_job_directory_durably(tmp_path):
    _, spool, ledger, maintenance = components(tmp_path)
    expired = cancel_job(ledger, finished_at=10.0)
    boundary = cancel_job(ledger, finished_at=50.0)
    recent = cancel_job(ledger, finished_at=80.0)
    expired_marker = create_job_marker(spool, expired["job_id"])
    boundary_marker = create_job_marker(spool, boundary["job_id"])
    recent_marker = create_job_marker(spool, recent["job_id"])

    result = maintenance.run_once(now=100.0)

    assert result.cutoff == 50.0
    assert result.deleted_jobs == (expired["job_id"],)
    assert result.removed_job_directories == (expired["job_id"],)
    assert result.missing_job_directories == ()
    assert result.pending_job_directories == ()
    assert ledger.get_job(expired["job_id"]) is None
    assert not Path(expired_marker).exists()
    assert ledger.get_job(boundary["job_id"]) is not None
    assert ledger.get_job(recent["job_id"]) is not None
    assert Path(boundary_marker).is_file()
    assert Path(recent_marker).is_file()


def test_run_once_deletes_ledger_row_before_touching_job_directory():
    job_id = str(uuid.uuid4())
    events = []

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            events.append(("tickets", now))
            return 2

        def delete_terminal_jobs_before(self, cutoff):
            events.append(("ledger", cutoff))
            return [job_id]

    class SpoolDouble:
        def job_directory(self, value):
            assert value == job_id
            return f"/server-owned/jobs/{value}"

        def remove(self, path):
            events.append(("filesystem", path))
            return True

    maintenance = MaintenanceService(
        SimpleNamespace(retention_seconds=10),
        LedgerDouble(),
        SpoolDouble(),
        logger=RecordingLogger(),
    )

    result = maintenance.run_once(now=100.0)

    assert events == [
        ("tickets", 100.0),
        ("ledger", 90.0),
        ("filesystem", f"/server-owned/jobs/{job_id}"),
    ]
    assert result.expired_tickets == 2


def test_missing_job_directory_is_successful_idempotent_cleanup(tmp_path):
    _, spool, ledger, maintenance = components(tmp_path)
    expired = cancel_job(ledger, finished_at=10.0)
    assert not Path(spool.job_directory(expired["job_id"])).exists()

    result = maintenance.run_once(now=100.0)

    assert result.deleted_jobs == (expired["job_id"],)
    assert result.removed_job_directories == ()
    assert result.missing_job_directories == (expired["job_id"],)
    assert result.failed_job_directories == ()
    assert result.pending_job_directories == ()


def test_failed_directory_cleanup_is_retained_for_next_in_process_pass():
    job_id = str(uuid.uuid4())

    class LedgerDouble:
        calls = 0

        def delete_expired_tickets(self, *, now):
            return 0

        def delete_terminal_jobs_before(self, cutoff):
            self.calls += 1
            return [job_id] if self.calls == 1 else []

    class SpoolDouble:
        calls = 0

        def job_directory(self, value):
            return f"/server-owned/jobs/{value}"

        def remove(self, path):
            self.calls += 1
            if self.calls == 1:
                raise OSError("injected removal failure")
            return True

    logger = RecordingLogger()
    maintenance = MaintenanceService(
        SimpleNamespace(retention_seconds=10),
        LedgerDouble(),
        SpoolDouble(),
        logger=logger,
    )

    failed = maintenance.run_once(now=100.0)
    retried = maintenance.run_once(now=101.0)

    assert failed.failed_job_directories == (job_id,)
    assert failed.pending_job_directories == (job_id,)
    assert retried.deleted_jobs == ()
    assert retried.removed_job_directories == (job_id,)
    assert retried.pending_job_directories == ()
    assert logger.events[0][0] == "flight.maintenance.job_directory_failed"


def test_expired_tickets_are_removed_without_removing_unexpired_ticket(tmp_path):
    _, _, ledger, maintenance = components(tmp_path)
    job = create_job(ledger, operation="predict", now=1.0)
    running = seal_queue_claim(ledger, job["job_id"])
    ledger.publish_outputs(
        job["job_id"],
        running["attempt"],
        [{
            "ordinal": 0,
            "rows": 0,
            "batches": 0,
            "bytes": 128,
            "sha256": DIGEST,
            "schema_fingerprint": "b" * 64,
            "relative_path": (
                f"spool/jobs/{job['job_id']}/attempts/1/outputs/0.arrow"
            ),
        }],
        result={"outputs": [0]},
        now=80.0,
    )
    expired_ticket, _ = ledger.issue_ticket(
        job_id=job["job_id"],
        ordinal=0,
        owner_subject="inventory",
        ttl_seconds=5,
        now=90.0,
    )
    active_ticket, _ = ledger.issue_ticket(
        job_id=job["job_id"],
        ordinal=0,
        owner_subject="inventory",
        ttl_seconds=20,
        now=90.0,
    )

    result = maintenance.run_once(now=100.0)

    assert result.expired_tickets == 1
    with ledger.connection() as connection:
        remaining = connection.scalar(
            select(func.count()).select_from(OutputTicket)
        )
    assert remaining == 1
    assert ledger.resolve_ticket(
        active_ticket,
        owner_subject="inventory",
        now=100.0,
    )["job_id"] == job["job_id"]
    assert expired_ticket != active_ticket


def test_model_generation_and_producing_job_are_never_retained_away(tmp_path):
    _, spool, ledger, maintenance = components(tmp_path)
    job = create_job(ledger, operation="fit", now=1.0)
    running = seal_queue_claim(ledger, job["job_id"])
    model_ref = "mdl_retained"
    checkpoint_path = spool.model_checkpoint_path(model_ref)
    metadata_path = spool.model_metadata_path(model_ref)
    spool.atomic_write_bytes(checkpoint_path, b"checkpoint")
    spool.atomic_write_json(metadata_path, {"modelRef": model_ref})
    job_marker = create_job_marker(spool, job["job_id"])
    ledger.publish_model(
        job["job_id"],
        running["attempt"],
        model_ref=model_ref,
        label="daily-model",
        generation=None,
        checkpoint_path=spool.model_relative_path(checkpoint_path),
        metadata_path=spool.model_relative_path(metadata_path),
        sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        metadata={"modelRef": model_ref},
        result={"modelRef": model_ref},
        now=10.0,
    )

    result = maintenance.run_once(now=100.0)

    assert result.deleted_jobs == ()
    assert ledger.get_job(job["job_id"])["state"] == JobState.SUCCEEDED.value
    assert ledger.get_model(model_ref)["producing_job_id"] == job["job_id"]
    assert Path(checkpoint_path).is_file()
    assert Path(metadata_path).is_file()
    assert Path(job_marker).is_file()


def test_start_and_shutdown_run_periodic_maintenance_once():
    called = threading.Event()

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            called.set()
            return 0

        def delete_terminal_jobs_before(self, cutoff):
            return []

    maintenance = MaintenanceService(
        SimpleNamespace(retention_seconds=10),
        LedgerDouble(),
        SimpleNamespace(),
        interval_seconds=0.01,
        logger=RecordingLogger(),
    )

    assert maintenance.start() is maintenance
    assert maintenance.start() is maintenance
    assert called.wait(1.0)
    assert maintenance.running is True

    maintenance.shutdown(timeout=1.0)
    maintenance.shutdown(timeout=1.0)

    assert maintenance.running is False


def test_shutdown_never_returns_while_maintenance_can_still_mutate():
    entered = threading.Event()
    release = threading.Event()

    class LedgerDouble:
        def delete_expired_tickets(self, *, now):
            entered.set()
            release.wait(1.0)
            return 0

        def delete_terminal_jobs_before(self, cutoff):
            return []

    maintenance = MaintenanceService(
        SimpleNamespace(retention_seconds=10),
        LedgerDouble(),
        SimpleNamespace(),
        interval_seconds=60,
        logger=RecordingLogger(),
    ).start()
    assert entered.wait(1.0)
    release_timer = threading.Timer(0.05, release.set)
    release_timer.start()

    try:
        started = time.monotonic()
        maintenance.shutdown(timeout=0.001)
    finally:
        release_timer.join(timeout=1.0)

    assert not release_timer.is_alive()
    assert time.monotonic() - started >= 0.02
    assert maintenance.running is False
