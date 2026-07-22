import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pyarrow.flight as flight
import pytest

from app.config import PROJECT_ROOT
from app.flight.config import FlightServiceConfig
from app.flight.constants import CAPABILITIES_ACTION, CONTRACT_NAME
from app.flight.contract import encode_document, response_document
from app.flight.observability import OperationalMetrics
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool
from app.flight.worker import WorkerPool


class RecordingLogger:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def event(self, event, **fields):
        with self._lock:
            self.events.append((event, fields))


class CapabilityCoordinator:
    def dispatch(self, _action, _owner, request, _document):
        return encode_document(response_document(request["request_id"], ready=True))


def _call_options(token="secret"):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode("ascii"))]
    )


def _action_body(request_id):
    return json.dumps({
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": request_id,
    }).encode("utf-8")


def test_action_and_rpc_logs_have_correlation_status_and_latency_without_secret(
    tmp_path,
):
    logger = RecordingLogger()
    metrics = OperationalMetrics()
    server = TransformerFlightServer(
        FlightServiceConfig(
            state_dir=str(tmp_path),
            port=0,
            profile="development",
            allow_plaintext=True,
        ),
        CapabilityCoordinator(),
        {"secret": "inventory"},
        metrics=metrics,
        logger=logger,
    )
    client = flight.FlightClient(("localhost", server.port))
    request_id = str(uuid.uuid4())
    try:
        list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, _action_body(request_id)),
            options=_call_options(),
        ))
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(client.do_action(
                flight.Action(CAPABILITIES_ACTION, _action_body(str(uuid.uuid4()))),
                options=_call_options("wrong-secret"),
            ))
    finally:
        server.shutdown()

    action = next(
        fields
        for event, fields in logger.events
        if event == "flight.action.completed" and fields.get("requestId") == request_id
    )
    assert action["action"] == CAPABILITIES_ACTION
    assert action["status"] == "OK"
    assert action["latencyMs"] >= 0

    rpc = [
        fields for event, fields in logger.events
        if event == "flight.rpc.completed"
    ]
    assert {fields["status"] for fields in rpc} >= {"OK", "UNAUTHENTICATED"}
    assert all(fields["latencyMs"] >= 0 for fields in rpc)
    assert "secret" not in json.dumps(logger.events)

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["rpcRequests"] == 2
    assert sum(snapshot["rpc"].values()) == 2


def test_worker_queue_metrics_are_aggregate_and_transition_log_is_correlated():
    job_id = str(uuid.uuid4())
    claimed = {
        "job_id": job_id,
        "attempt": 1,
        "selected_device": "cpu",
        "queued_at": 10.0,
        "started_at": 15.25,
    }

    class LedgerDouble:
        def claim_next_job(self, selected_device, *, worker_id):
            assert selected_device == "cpu"
            assert worker_id == "worker-1"
            return claimed

    metrics = OperationalMetrics()
    logger = RecordingLogger()
    pool = WorkerPool(
        SimpleNamespace(
            cpu_capacity=1,
            cuda_capacity=1,
            shutdown_drain_seconds=1.0,
            cancel_grace_seconds=0.1,
        ),
        LedgerDouble(),
        object(),
        metrics=metrics,
        logger=logger,
    )
    executed = []
    pool._execute_claimed = executed.append

    assert pool.run_once("cpu", worker_id="worker-1") is True
    assert executed == [claimed]

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["jobsStarted"] == 1
    assert snapshot["counters"]["workerQueueWaitSeconds"] == 5.25
    assert snapshot["counters"]["jobTransitions.QUEUED.RUNNING"] == 1
    assert job_id not in json.dumps(snapshot)

    event, fields = logger.events[0]
    assert event == "flight.job.transition"
    assert fields == {
        "jobId": job_id,
        "attempt": 1,
        "device": "cpu",
        "fromState": "QUEUED",
        "toState": "RUNNING",
        "queueWaitSeconds": 5.25,
    }


def test_service_sigterm_drains_cleanly_after_signal_handlers_are_installed(tmp_path):
    state_dir = tmp_path / "state"
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"secret": "inventory"}), encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(PROJECT_ROOT) / "app" / "main.py"),
            "flight",
            "serve",
            "--port", "0",
            "--profile", "development",
            "--allow-plaintext",
            "--bearer-tokens-file", str(tokens),
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={
            **os.environ,
            "TRANSFORMER_STATE_DIR": str(state_dir),
        },
    )
    lines = queue.Queue()

    def drain_stderr():
        for line in process.stderr:
            lines.put(line)

    pump = threading.Thread(target=drain_stderr, daemon=True)
    pump.start()
    events = []
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            event = json.loads(line)
            events.append(event)
            if event.get("event") == "flight.service.serving":
                break
        assert any(
            event.get("event") == "flight.service.serving" for event in events
        ), events

        process.send_signal(signal.SIGTERM)
        try:
            return_code = process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            while not lines.empty():
                events.append(json.loads(lines.get_nowait()))
            pytest.fail(f"Flight service did not stop after SIGTERM: {events!r}")
        assert return_code == 0
        pump.join(timeout=2.0)
        while not lines.empty():
            events.append(json.loads(lines.get_nowait()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

    names = [event.get("event") for event in events]
    assert "flight.service.signal" in names
    assert "flight.service.draining" in names
    assert "flight.service.stopped" in names
    assert names.index("flight.service.serving") < names.index("flight.service.signal")
    assert names.index("flight.service.signal") < names.index("flight.service.stopped")

    replacement = Spool(state_dir).initialize()
    replacement.acquire_lock()
    replacement.release_lock()
