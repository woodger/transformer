from __future__ import annotations

import hashlib
import json
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from app.flight.arrow import schema_fingerprint
from app.flight.config import FlightServiceConfig
from app.flight.maintenance import MaintenanceService
from app.flight.observability import OperationalMetrics
from app.flight.spool import Spool
from app.flight.worker_plan import WorkerPlanBuilder, WorkerPlanError
from app.service.adapters.outbound.artifact_storage.recovery_store import (
    RecoveryStore,
)
from app.service.application.services.worker_pool import WorkerPool
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import ModelArtifactRecord
from tests.flight_v4_helpers import (
    DATA_CONTRACT_SHA256,
    close_input,
    create_fit,
    create_predict,
    internal_data_contract,
)


class RecordingLogger:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))


def _config(tmp_path):
    return FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        allow_plaintext=True,
        cpu_capacity=1,
        subprocess_timeout_seconds=5,
    ).validate()


def _stores(tmp_path, ledger):
    config = _config(tmp_path)
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    recovery = RecoveryStore(tmp_path / "recovery").initialize()
    builder = WorkerPlanBuilder(
        config,
        ledger,
        spool,
        recovery,
        python_executable=sys.executable,
    )
    return config, spool, recovery, builder


def _fit_table(value):
    schema = pa.schema([
        pa.field("src", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("tgt", pa.list_(pa.float32(), 6), nullable=False),
    ])
    return pa.Table.from_arrays(
        [
            pa.array([[value, value + 1, value + 2, value + 3]],
                     type=schema.field("src").type),
            pa.array([[0.0, 0.0, 0.0, 0.0, 0.2, 1.0]],
                     type=schema.field("tgt").type),
        ],
        schema=schema,
    )


def _commit_real_input(ledger, recovery, job, ordinal, *, value=1.0):
    payload_id = str(uuid.uuid4())
    upload_token = uuid.uuid4().hex
    destination = recovery.input_candidate_path(
        job["job_id"],
        ordinal,
        payload_id,
        upload_token,
    )
    recovery.ensure_parent(destination)
    table = _fit_table(value)
    with pa.OSFile(destination, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    raw = Path(destination).read_bytes()
    relative = recovery.relative_path(destination)
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=ordinal,
        client_execution_id=job["client_execution_id"],
        fencing_token=job["fencing_token"],
        upload_token=upload_token,
        candidate_path=relative,
        storage_class="recovery",
    )
    return ledger.commit_input(
        upload_token=upload_token,
        job_id=job["job_id"],
        client_execution_id=job["client_execution_id"],
        fencing_token=job["fencing_token"],
        relative_path=relative,
        schema_id="inventory.sequence.fit.v2",
        data_contract_sha256=DATA_CONTRACT_SHA256,
        rows=1,
        batches=1,
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_fingerprint=schema_fingerprint(table.schema),
        source_width=4,
        feature_dim=2,
        selected_device="cpu",
        max_payloads=100,
        max_job_bytes=1024 * 1024,
        storage_class="recovery",
    )


def test_open_fit_plan_is_an_immutable_worker_v3_snapshot(
    tmp_path,
    postgres_ledger,
):
    _, _, recovery, builder = _stores(tmp_path, postgres_ledger)
    job = create_fit(postgres_ledger)
    _commit_real_input(postgres_ledger, recovery, job, 0)
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")
    assert running is not None
    assert running.input_state is InputState.OPEN

    plan = builder.build(running, running.attempt)
    document = json.loads(Path(plan.manifest_path).read_text())

    assert plan.protocol_version == 3
    assert plan.argv[:3] == (
        sys.executable,
        "-m",
        "app.worker.bootstrap",
    )
    assert document["inputClosed"] is False
    assert document["inputRevision"] == 1
    assert [item["ordinal"] for item in document["inputs"]] == [0]
    assert document["inputs"][0]["artifact"]["path"] == plan.inputs[0].absolute_path
    assert "clientExecutionId" not in document
    assert "fencingToken" not in document
    assert document["attemptId"] == running.attempt_id


def test_plan_rejects_tampered_durable_input(tmp_path, postgres_ledger):
    _, _, recovery, builder = _stores(tmp_path, postgres_ledger)
    job = create_fit(postgres_ledger)
    committed = _commit_real_input(postgres_ledger, recovery, job, 0)
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")
    path = recovery.absolute_path(committed["relative_path"])
    Path(path).write_bytes(b"tampered")

    with pytest.raises(WorkerPlanError) as error:
        builder.build(running, running.attempt)

    assert error.value.code is ErrorCode.RECOVERY_INPUT_UNAVAILABLE


def test_worker_snapshot_and_updates_follow_contiguous_ordinal_prefix(
    tmp_path,
    postgres_ledger,
):
    _, _, recovery, builder = _stores(tmp_path, postgres_ledger)
    job = create_fit(postgres_ledger)
    _commit_real_input(postgres_ledger, recovery, job, 1, value=10.0)
    waiting = postgres_ledger.get_execution_job(job["job_id"])
    assert waiting.execution_state is ExecutionState.WAITING_INPUT
    assert waiting.input_frame_count == 0

    _commit_real_input(postgres_ledger, recovery, job, 0, value=1.0)
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")
    assert running.input_frame_count == 2
    assert [item.ordinal for item in builder.build(
        running,
        running.attempt,
    ).inputs] == [0, 1]

    _commit_real_input(postgres_ledger, recovery, job, 3, value=30.0)
    still_gapped = postgres_ledger.get_execution_job(job["job_id"])
    assert still_gapped.input_frame_count == 2
    assert builder.streaming_inputs(still_gapped, 2) == ()

    _commit_real_input(postgres_ledger, recovery, job, 2, value=20.0)
    advanced = postgres_ledger.get_execution_job(job["job_id"])
    assert advanced.input_frame_count == 4
    assert [item.ordinal for item in builder.streaming_inputs(
        advanced,
        2,
    )] == [2, 3]


def test_worker_pool_claims_fifo_and_forwards_input_notifications(
    tmp_path,
    postgres_ledger,
):
    config, _, recovery, _ = _stores(tmp_path, postgres_ledger)
    jobs = [create_fit(postgres_ledger) for _ in range(2)]
    for ordinal, job in enumerate(jobs):
        _commit_real_input(
            postgres_ledger,
            recovery,
            job,
            0,
            value=float(ordinal),
        )

    executed = []
    input_notifications = []
    executor = SimpleNamespace(
        execute=executed.append,
        notify_cancel=lambda _job_id: None,
        notify_input=input_notifications.append,
        interrupt_for_shutdown=lambda: None,
    )
    inventory = SimpleNamespace(
        schedulable_devices=lambda: (),
        snapshot=lambda: SimpleNamespace(devices=()),
    )
    pool = WorkerPool(
        config,
        postgres_ledger,
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
        device_inventory=inventory,
        attempt_executor=executor,
    )

    pool.notify_input(jobs[0]["job_id"])
    assert pool.run_once("cpu", worker_id="worker-1") is True
    assert pool.run_once("cpu", worker_id="worker-1") is True

    assert input_notifications == [jobs[0]["job_id"]]
    assert [job.job_id for job in executed] == [job["job_id"] for job in jobs]
    assert all(job.execution_state is ExecutionState.RUNNING for job in executed)


def test_worker_pool_recovers_a_lost_queue_notification(
    tmp_path,
    postgres_ledger,
):
    config, spool, recovery, _ = _stores(tmp_path, postgres_ledger)
    executed = threading.Event()
    executor = SimpleNamespace(
        execute=lambda _job: executed.set(),
        notify_cancel=lambda _job_id: None,
        notify_input=lambda _job_id: None,
        interrupt_for_shutdown=lambda: None,
    )
    inventory = SimpleNamespace(
        schedulable_devices=lambda: (),
        snapshot=lambda: SimpleNamespace(devices=()),
    )
    pool = WorkerPool(
        config,
        postgres_ledger,
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
        device_inventory=inventory,
        attempt_executor=executor,
    ).start()
    maintenance = MaintenanceService(
        config,
        postgres_ledger,
        spool,
        interval_seconds=60,
        queue_reconciler=pool.notify_queued,
        logger=RecordingLogger(),
        metrics=OperationalMetrics(),
    )
    try:
        job = create_fit(postgres_ledger)
        _commit_real_input(postgres_ledger, recovery, job, 0)
        maintenance.run_once()

        assert executed.wait(0.5)
    finally:
        pool.shutdown(timeout=0.1)


class _ModelArtifactLedger:
    def __init__(self, ledger, artifact):
        self._ledger = ledger
        self._artifact = artifact

    def __getattr__(self, name):
        return getattr(self._ledger, name)

    def get_model_artifact(self, *_args, **_kwargs):
        return self._artifact


@pytest.mark.parametrize(
    ("condition", "expected_code"),
    [
        ("unknown", ErrorCode.NOT_FOUND),
        ("unavailable", ErrorCode.MODEL_UNAVAILABLE),
        ("corrupt", ErrorCode.MODEL_CORRUPT),
    ],
)
def test_predict_plan_preserves_stable_model_lifecycle_errors(
    tmp_path,
    postgres_ledger,
    condition,
    expected_code,
):
    config, spool, recovery, _ = _stores(tmp_path, postgres_ledger)
    job = create_predict(postgres_ledger)
    close_input(postgres_ledger, job)
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")
    assert running is not None

    artifact = None
    if condition != "unknown":
        checkpoint = spool.model_checkpoint_path("mdl_seed")
        if condition == "corrupt":
            spool.atomic_write_bytes(checkpoint, b"corrupt checkpoint")
        artifact = ModelArtifactRecord(
            model_ref="mdl_seed",
            owner_subject=running.owner_subject,
            checkpoint_path=spool.model_relative_path(checkpoint),
            byte_count=(
                len(b"corrupt checkpoint")
                if condition == "corrupt"
                else 1
            ),
            sha256="0" * 64,
            data_contract=internal_data_contract(),
            ml_contract=running.ml_contract,
            objective_config_sha256=running.ml_contract[
                "objectiveConfigSha256"
            ],
        )
    builder = WorkerPlanBuilder(
        config,
        _ModelArtifactLedger(postgres_ledger, artifact),
        spool,
        recovery,
        python_executable=sys.executable,
    )

    with pytest.raises(WorkerPlanError) as error:
        builder.build(running, running.attempt)

    assert error.value.code is expected_code
