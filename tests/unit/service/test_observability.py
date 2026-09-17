import json
import threading
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pyarrow.flight as flight
import pytest

from app.contracts.semantic.v3 import ModelContract
from app.contracts.worker.v14.model_config import ModelConfig
from app.contracts.worker.v14.model_definition import resolved_semantic_digests
from app.service.adapters.inbound.flight.constants import (
    CAPABILITIES_ACTION,
)
from app.service.adapters.inbound.flight.documents import (
    encode_document,
    response_document,
)
from app.service.adapters.inbound.flight.server import TransformerFlightServer
from app.service.adapters.observability import OperationalMetrics
from app.service.application.services.worker_pool import WorkerPool
from app.service.bootstrap.config import FlightServiceConfig
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import ExecutionJobRecord
from tests.fixture_documents import semantic_fixture_document
from tests.support.authentication import StaticAccessTokenAuthenticator


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
        headers=[(b"authorization", f"Bearer {token}".encode("ascii"))],
        timeout=5.0,
    )


def _action_body(request_id):
    return json.dumps({
        "requestId": request_id,
    }).encode("utf-8")


def test_action_and_rpc_logs_have_correlation_status_and_latency_without_secret(
    tmp_path,
):
    logger = RecordingLogger()
    metrics = OperationalMetrics()
    server = TransformerFlightServer(
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "runtime"),
            port=0,
        ),
        CapabilityCoordinator(),
        StaticAccessTokenAuthenticator({"secret": "inventory"}),
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
        client.close()
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
    contract = ModelContract.from_document(
        semantic_fixture_document("single-regression")["modelContract"],
    )
    data_contract = {
        "identity": "test.dataset",
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": "d" * 64,
        "seqLen": 2,
        "featureDim": 2,
    }
    claimed = ExecutionJobRecord(
        job_id=job_id,
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.OPEN,
        execution_state=ExecutionState.RUNNING,
        input_revision=1,
        selected_device="cpu",
        model_label="model",
        input_model_ref=None,
        prediction_column="predictions",
        source_encoding={
            "featureBlocks": [
                {"windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        model_config=None,
        training_config=None,
        data_contract=data_contract,
        model_contract=contract.to_document(),
        semantic_digests=resolved_semantic_digests(
            contract,
            "d" * 64,
            ModelConfig.from_tuning(
                contract.model_tuning,
                seq_len=2,
                feature_dim=2,
            ),
        ),
        config_hash="a" * 64,
        manifest_sha256=None,
        feature_dim=2,
        input_frame_count=1,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=10.0,
        started_at=15.25,
    )
    queued = replace(
        claimed,
        execution_state=ExecutionState.QUEUED,
        attempt=0,
        started_at=None,
    )

    class LedgerDouble:
        def queued_execution_jobs(self):
            return [queued]

        def get_execution_job(self, requested_job_id):
            assert requested_job_id == job_id
            return queued

        def claim_execution_job(
            self,
            requested_job_id,
            selected_device,
            *,
            worker_id,
            device_id,
        ):
            assert requested_job_id == job_id
            assert selected_device == "cpu"
            assert worker_id == "worker-1"
            assert device_id is None
            return claimed

    metrics = OperationalMetrics()
    logger = RecordingLogger()
    executed = []
    attempt_executor = SimpleNamespace(
        execute=executed.append,
        notify_cancel=lambda _job_id: None,
        notify_input=lambda _job_id: None,
        interrupt_for_shutdown=lambda: None,
    )
    device_inventory = SimpleNamespace(
        schedulable_devices=lambda: (),
        snapshot=lambda: SimpleNamespace(devices=()),
    )
    pool = WorkerPool(
        SimpleNamespace(
            cpu_capacity=1,
            shutdown_drain_seconds=1.0,
            cancel_grace_seconds=0.1,
        ),
        LedgerDouble(),
        metrics=metrics,
        logger=logger,
        device_inventory=device_inventory,
        attempt_executor=attempt_executor,
    )

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
        "deviceId": None,
        "fromState": "QUEUED",
        "toState": "RUNNING",
        "queueWaitSeconds": 5.25,
    }
