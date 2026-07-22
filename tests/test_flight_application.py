import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pyarrow.flight as flight
import pytest

import app.flight.application as application_module
from app.cli.help import build_parser
from app.flight.application import FlightApplication
from app.flight.config import FlightServiceConfig
from app.flight.constants import CAPABILITIES_ACTION, CONTRACT_NAME, ErrorCode, JobState
from app.flight.ledger import Ledger
from app.flight.process import capture_worker_process
from app.flight.spool import Spool, StateDirectoryLocked


def options():
    return flight.FlightCallOptions(
        headers=[(b"authorization", b"Bearer secret")]
    )


def config(tmp_path):
    return FlightServiceConfig(
        state_dir=str(tmp_path),
        port=0,
        profile="development",
        allow_plaintext=True,
        disk_min_free_bytes=1,
    ).validate()


def test_application_is_runnable_and_owns_state_directory(tmp_path):
    application = FlightApplication.build(
        config(tmp_path),
        bearer_tokens={"secret": "inventory"},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    body = json.dumps({
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": str(uuid.uuid4()),
    }).encode()
    try:
        assert application.maintenance.running is True
        result = list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, body),
            options=options(),
        ))
        assert json.loads(result[0].body.to_pybytes())["protocolVersions"] == [1]

        with pytest.raises(StateDirectoryLocked):
            FlightApplication.build(
                config(tmp_path),
                bearer_tokens={"secret": "inventory"},
            )
    finally:
        application.shutdown()

    assert application.maintenance.running is False

    replacement = FlightApplication.build(
        config(tmp_path),
        bearer_tokens={"secret": "inventory"},
    )
    replacement.shutdown()


def test_build_failure_after_server_construction_releases_all_resources(
    tmp_path,
    monkeypatch,
):
    created_servers = []

    class FakeServer:
        port = 12345

        def __init__(self, *args, **kwargs):
            self.shutdown_called = False
            created_servers.append(self)

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(application_module, "TransformerFlightServer", FakeServer)
    monkeypatch.setattr(
        application_module,
        "MaintenanceService",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected maintenance construction failure")
        ),
    )

    with pytest.raises(RuntimeError, match="maintenance construction failure"):
        FlightApplication.build(
            config(tmp_path),
            bearer_tokens={"secret": "inventory"},
        )

    assert created_servers and created_servers[0].shutdown_called is True
    replacement_lock = Spool(tmp_path).initialize()
    replacement_lock.acquire_lock()
    replacement_lock.release_lock()


def test_build_start_failure_stops_started_components_and_releases_lock(
    tmp_path,
    monkeypatch,
):
    events = []

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("worker-start")

        def shutdown(self, timeout):
            events.append(("worker-shutdown", timeout))

        def notify_cancel(self, job_id):
            pass

        def notify_queued(self, job_id=None):
            pass

    class FakeMaintenance:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("maintenance-start")
            raise RuntimeError("injected maintenance start failure")

        def shutdown(self, timeout):
            events.append(("maintenance-shutdown", timeout))

    class FakeServer:
        port = 12345

        def __init__(self, *args, **kwargs):
            pass

        def shutdown(self):
            events.append("server-shutdown")

    monkeypatch.setattr(application_module, "WorkerPool", FakeWorker)
    monkeypatch.setattr(application_module, "MaintenanceService", FakeMaintenance)
    monkeypatch.setattr(application_module, "TransformerFlightServer", FakeServer)

    with pytest.raises(RuntimeError, match="maintenance start failure"):
        FlightApplication.build(
            config(tmp_path),
            bearer_tokens={"secret": "inventory"},
        )

    assert events == [
        "worker-start",
        "maintenance-start",
        "server-shutdown",
        ("worker-shutdown", 0),
        ("maintenance-shutdown", 0),
    ]
    replacement_lock = Spool(tmp_path).initialize()
    replacement_lock.acquire_lock()
    replacement_lock.release_lock()


def test_shutdown_continues_after_server_error_before_releasing_state_lock(
    tmp_path,
    monkeypatch,
):
    application = FlightApplication.build(
        config(tmp_path),
        bearer_tokens={"secret": "inventory"},
    )
    real_shutdown = application.server.shutdown

    def stop_then_fail():
        real_shutdown()
        raise RuntimeError("injected server shutdown failure")

    monkeypatch.setattr(application.server, "shutdown", stop_then_fail)

    with pytest.raises(RuntimeError, match="server shutdown failure"):
        application.shutdown()

    assert application.maintenance.running is False
    assert all(not thread.is_alive() for thread in application.worker._threads)
    replacement_lock = Spool(tmp_path).initialize()
    replacement_lock.acquire_lock()
    replacement_lock.release_lock()


def test_shutdown_closes_queue_claims_before_stopping_flight_server():
    events = []

    class Worker:
        def stop_claiming(self):
            events.append("worker-stop-claiming")

        def shutdown(self, timeout):
            events.append(("worker-shutdown", timeout))

    class Coordinator:
        def set_draining(self, value):
            events.append(("coordinator-draining", value))

    class Component:
        def __init__(self, event):
            self.event = event

        def shutdown(self, *args):
            events.append((self.event, *args))

        def close(self):
            events.append((self.event,))

        def release_lock(self):
            events.append((self.event,))

    application = FlightApplication(
        SimpleNamespace(shutdown_drain_seconds=3),
        Component("spool-release"),
        Component("ledger-close"),
        Coordinator(),
        Component("server-shutdown"),
        Component("maintenance-shutdown"),
        Worker(),
        object(),
        SimpleNamespace(event=lambda *args, **kwargs: None),
    )

    application.shutdown()

    assert events[:3] == [
        "worker-stop-claiming",
        ("coordinator-draining", True),
        ("server-shutdown",),
    ]
    assert events[3:] == [
        ("worker-shutdown", 3),
        ("maintenance-shutdown", 3),
        ("ledger-close",),
        ("spool-release",),
    ]


def test_restart_recovers_nonterminal_states_without_retrying_running_job(
    tmp_path,
    monkeypatch,
):
    service_config = config(tmp_path)
    spool = Spool(service_config.state_dir).initialize()
    ledger = Ledger(service_config.database_path).initialize()

    def create(label):
        return ledger.create_job(
            job_id=str(uuid.uuid4()),
            owner_subject="inventory",
            operation="fit",
            requested_device="cpu",
            prediction_column="out",
            config_hash="a" * 64,
            model_label=label,
        )

    def seal_and_queue(job):
        ledger.seal_job(job["job_id"], manifest_hash="b" * 64, manifest=[])
        ledger.queue_job(job["job_id"], selected_device="cpu")

    uploading = create("uploading")
    sealed = create("sealed")
    ledger.seal_job(sealed["job_id"], manifest_hash="b" * 64, manifest=[])

    interrupted = create("interrupted")
    seal_and_queue(interrupted)
    interrupted = ledger.claim_next_job("cpu")
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    orphan_identity = capture_worker_process(orphan.pid)
    ledger.set_attempt_process(
        interrupted["job_id"],
        interrupted["attempt"],
        pid=orphan.pid,
        pgid=orphan.pid,
        boot_id=orphan_identity.boot_id,
        process_start_ticks=orphan_identity.start_ticks,
    )
    cancelling = create("cancelling")
    seal_and_queue(cancelling)
    cancelling = ledger.claim_next_job("cpu")
    ledger.transition_job(cancelling["job_id"], JobState.CANCELLING)
    queued = create("queued")
    seal_and_queue(queued)

    for job in (interrupted, cancelling):
        spool.atomic_write_bytes(
            spool.attempt_checkpoint_path(job["job_id"], job["attempt"]),
            b"unpublished checkpoint",
        )
        spool.atomic_write_bytes(
            spool.attempt_output_path(job["job_id"], job["attempt"], 0),
            b"unpublished output",
        )
        spool.atomic_write_bytes(
            spool.attempt_stderr_path(job["job_id"], job["attempt"]),
            b"diagnostic",
        )

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return self

        def shutdown(self, timeout):
            pass

        def notify_cancel(self, job_id):
            pass

        def notify_queued(self, job_id=None):
            pass

    class FakeMaintenance:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return self

        def shutdown(self, timeout):
            pass

    class FakeServer:
        port = 12345

        def __init__(self, *args, **kwargs):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(application_module, "WorkerPool", FakeWorker)
    monkeypatch.setattr(application_module, "MaintenanceService", FakeMaintenance)
    monkeypatch.setattr(application_module, "TransformerFlightServer", FakeServer)

    application = None
    try:
        application = FlightApplication.build(
            service_config,
            bearer_tokens={"secret": "inventory"},
        )
        recovered = application.ledger
        assert recovered.get_job(uploading["job_id"])["state"] == JobState.UPLOADING.value
        assert recovered.get_job(sealed["job_id"])["state"] == JobState.SEALED.value
        assert recovered.get_job(queued["job_id"])["state"] == JobState.QUEUED.value
        failed = recovered.get_job(interrupted["job_id"])
        assert failed["state"] == JobState.FAILED.value
        assert failed["attempt"] == 1
        assert failed["error_code"] == ErrorCode.EXECUTION_INTERRUPTED.value
        assert orphan.wait(timeout=5) < 0
        assert recovered.get_job(cancelling["job_id"])["state"] == JobState.CANCELLED.value
        assert [job["job_id"] for job in recovered.list_jobs([JobState.QUEUED])] == [
            queued["job_id"]
        ]
        for job in (interrupted, cancelling):
            assert not os.path.exists(
                spool.attempt_checkpoint_path(job["job_id"], job["attempt"])
            )
            assert not os.path.exists(
                spool.attempt_output_path(job["job_id"], job["attempt"], 0)
            )
            assert os.path.isfile(
                spool.attempt_stderr_path(job["job_id"], job["attempt"])
            )
    finally:
        if application is not None:
            application.shutdown()
        if orphan.poll() is None:
            os.killpg(orphan.pid, signal.SIGKILL)
            orphan.wait()


def test_flight_serve_cli_contains_only_service_configuration():
    args = build_parser().parse_args([
        "flight",
        "serve",
        "--state-dir", "/var/lib/transformer",
        "--host", "127.0.0.1",
        "--port", "8815",
        "--profile", "development",
        "--allow-plaintext",
        "--bearer-tokens-file", "/run/secrets/flight-tokens.json",
    ])

    assert args.action == "flight"
    assert args.flight_action == "serve"
    assert args.state_dir == "/var/lib/transformer"
    assert args.host == "127.0.0.1"
    assert args.port == 8815
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "device")

    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "flight",
            "serve",
            "--bind-host",
            "127.0.0.1",
        ])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve-flight"])
