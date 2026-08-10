import json
import os
import signal
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pyarrow.flight as flight
import pytest

import app.flight.application as application_module
from app.cli.help import build_parser
from app.database.tokens import AccessTokenStore
from app.flight.application import FlightApplication
from app.flight.config import FlightServiceConfig
from app.flight.constants import CAPABILITIES_ACTION, CONTRACT_NAME
from app.flight.process import capture_worker_process
from app.flight.spool import RuntimeDirectoryLocked, Spool
from app.service.domain.job import ExecutionState, InputState
from tests.flight_v3_helpers import commit_input, create_fit


@pytest.fixture(autouse=True)
def _use_postgres_test_schema(
    monkeypatch,
    postgres_config,
    postgres_database,
    tmp_path,
):
    monkeypatch.setattr(
        application_module,
        "load_database_config",
        lambda: postgres_config,
    )
    monkeypatch.setattr(
        FlightServiceConfig,
        "models_dir",
        property(lambda _self: str(tmp_path / "models")),
    )


def options(token="secret"):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode("ascii"))],
        timeout=5.0,
    )


def config(tmp_path):
    return FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        port=0,
        allow_plaintext=True,
    ).validate()


def test_application_is_runnable_and_owns_runtime_directory(tmp_path):
    application = FlightApplication.build(
        config(tmp_path),
        bearer_tokens={"secret": "inventory"},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    body = json.dumps({
        "contract": CONTRACT_NAME,
        "version": 3,
        "requestId": str(uuid.uuid4()),
    }).encode()
    try:
        assert application.maintenance.running is True
        result = list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, body),
            options=options(),
        ))
        assert json.loads(result[0].body.to_pybytes())["protocolVersions"] == [3]

        with pytest.raises(RuntimeDirectoryLocked):
            FlightApplication.build(
                config(tmp_path),
                bearer_tokens={"secret": "inventory"},
            )
    finally:
        client.close()
        application.shutdown()

    assert application.maintenance.running is False

    replacement = FlightApplication.build(
        config(tmp_path),
        bearer_tokens={"secret": "inventory"},
    )
    replacement.shutdown()


def test_application_uses_database_token_cache_without_query_per_rpc(
    tmp_path,
    monkeypatch,
    postgres_database,
):
    issued = AccessTokenStore(postgres_database).issue("inventory")
    credential_loads = []

    class CountingAccessTokenStore(AccessTokenStore):
        def active_credentials(self):
            credential_loads.append(True)
            return super().active_credentials()

    monkeypatch.setattr(
        application_module,
        "AccessTokenStore",
        CountingAccessTokenStore,
    )
    application = FlightApplication.build(config(tmp_path))
    client = flight.FlightClient(("localhost", application.server.port))
    try:
        for _ in range(2):
            body = json.dumps({
                "contract": CONTRACT_NAME,
                "version": 3,
                "requestId": str(uuid.uuid4()),
            }).encode()
            result = list(client.do_action(
                flight.Action(CAPABILITIES_ACTION, body),
                options=options(issued.token),
            ))
            assert json.loads(result[0].body.to_pybytes())["protocolVersions"] == [3]

        assert credential_loads == [True]
    finally:
        client.close()
        application.shutdown()


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

        def notify_input(self, job_id):
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


def test_restart_requeues_persistent_fit_and_recovers_other_states(
    tmp_path,
    monkeypatch,
    postgres_ledger,
    request,
):
    service_config = config(tmp_path)
    spool = Spool(service_config.runtime_dir, tmp_path / "models").initialize()
    ledger = postgres_ledger
    ledger.synchronize_runtime_epoch(spool.storage_epoch())

    waiting = create_fit(ledger)
    interrupted = create_fit(ledger)
    commit_input(ledger, interrupted, 0)
    interrupted = ledger.claim_next_job("cpu")
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )

    def stop_orphan():
        if orphan.poll() is None:
            try:
                os.killpg(orphan.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        orphan.wait(timeout=5)

    request.addfinalizer(stop_orphan)
    orphan_identity = capture_worker_process(orphan.pid)
    ledger.set_attempt_process(
        interrupted["job_id"],
        interrupted["attempt"],
        attempt_id=interrupted["attempt_id"],
        pid=orphan.pid,
        pgid=orphan.pid,
        boot_id=orphan_identity.boot_id,
        process_start_ticks=orphan_identity.start_ticks,
    )
    cancelling = create_fit(ledger)
    commit_input(ledger, cancelling, 0)
    cancelling = ledger.claim_next_job("cpu")
    ledger.transition_job(cancelling["job_id"], ExecutionState.CANCELLING)
    queued = create_fit(ledger)
    commit_input(ledger, queued, 0)

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

        def notify_input(self, job_id):
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
        waiting_job = recovered.get_job(waiting["job_id"])
        assert waiting_job["input_state"] == InputState.OPEN.value
        assert waiting_job["execution_state"] == ExecutionState.WAITING_INPUT.value
        assert recovered.get_job(queued["job_id"])["execution_state"] == ExecutionState.QUEUED.value
        retrying = recovered.get_job(interrupted["job_id"])
        assert retrying["execution_state"] == ExecutionState.RETRYING.value
        assert retrying["attempt"] == 1
        assert retrying["error_code"] is None
        assert orphan.wait(timeout=5) < 0
        assert recovered.get_job(cancelling["job_id"])["execution_state"] == ExecutionState.CANCELLED.value
        assert {
            job["job_id"]
            for job in recovered.list_jobs([
                ExecutionState.QUEUED,
                ExecutionState.RETRYING,
            ])
        } == {interrupted["job_id"], queued["job_id"]}
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


def test_flight_serve_cli_contains_only_service_configuration():
    omitted = build_parser().parse_args(["flight", "serve"])
    assert omitted.host is None
    assert omitted.port is None
    assert not hasattr(omitted, "config")
    assert not hasattr(omitted, "runtime_dir")

    args = build_parser().parse_args([
        "flight",
        "serve",
        "--host", "127.0.0.1",
        "--port", "8815",
        "--allow-plaintext",
    ])

    assert args.action == "flight"
    assert args.flight_action == "serve"
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
