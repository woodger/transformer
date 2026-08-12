from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.flight as flight

from app.contracts.worker.v3.config import TrainConfig, train_config_to_manifest
from app.contracts.worker.v3.objective import ml_contract
from app.service.adapters.inbound.flight.constants import (
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    STATUS_ACTION,
)
from app.service.bootstrap.application import FlightApplication
from app.service.bootstrap.config import FlightServiceConfig
from app.service.domain.input_manifest import manifest_sha256
from app.service.domain.job import ExecutionState, InputState

OWNER = "inventory"
TOKEN = "secret"
DATA_CONTRACT_SHA256 = "d" * 64

MODEL_CONFIG = {
    "seqLen": 2,
    "hidden": 4,
    "layers": 1,
    "dropout": 0.0,
    "nhead": 2,
    "mode": "relaxed",
}
TRAIN_CONFIG = TrainConfig(
    lr=0.001,
    weight_decay=0.0,
    batch_size=8,
    epochs=1,
    loss_stage=4,
    loss_schedule="none",
    stage_size=1,
    use_amp=False,
    seed=29,
    deterministic=True,
)
TRAINING_CONFIG = train_config_to_manifest(TRAIN_CONFIG)
ML_CONTRACT = ml_contract(TRAIN_CONFIG)
DATA_CONTRACT = {
    "id": "inventory.learning-dataset",
    "version": 1,
    "dataContractSha256": DATA_CONTRACT_SHA256,
    "seqLen": 2,
    "featureDim": 2,
    "targetSchemaId": "inventory.target.v1",
}


def _service_config(tmp_path):
    return FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        allow_plaintext=True,
        cpu_capacity=1,
        cancel_grace_seconds=0.1,
        shutdown_drain_seconds=1.0,
        maintenance_interval_seconds=60,
        subprocess_timeout_seconds=30.0,
    ).validate()


def _auth():
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {TOKEN}".encode())],
        timeout=5.0,
    )


def _action(client, name, **fields):
    request = {
        "contract": CONTRACT_NAME,
        "version": 4,
        "requestId": str(uuid.uuid4()),
        **fields,
    }
    results = list(client.do_action(
        flight.Action(name, json.dumps(request).encode()),
        options=_auth(),
    ))
    assert len(results) == 1
    return json.loads(results[0].body.to_pybytes())


def _fit_schema():
    return pa.schema([
        pa.field("src", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("tgt", pa.list_(pa.float32(), 6), nullable=False),
    ])


def _batch(source_rows, target_rows):
    schema = _fit_schema()
    return pa.RecordBatch.from_arrays(
        [
            pa.array(source_rows, type=schema.field("src").type),
            pa.array(target_rows, type=schema.field("tgt").type),
        ],
        schema=schema,
    )


def _put(client, created, ordinal, batches):
    payload_id = str(uuid.uuid4())
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v4", "jobs", created["jobId"], "inputs", str(ordinal)
    )
    writer, results = client.do_put(
        descriptor,
        _fit_schema(),
        options=_auth(),
    )
    rows = sum(batch.num_rows for batch in batches)
    writer.write_metadata(pa.py_buffer(json.dumps({
        "contract": CONTRACT_NAME,
        "version": 4,
        "jobId": created["jobId"],
        "clientExecutionId": created["ownership"]["clientExecutionId"],
        "fencingToken": created["ownership"]["fencingToken"],
        "payloadId": payload_id,
        "ordinal": ordinal,
        "schemaId": FIT_SCHEMA_ID,
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "rows": rows,
    }).encode()))
    for batch in batches:
        writer.write_batch(batch)
    writer.done_writing()
    result = json.loads(results.read().to_pybytes())
    assert results.read() is None
    writer.close()
    return result


def test_real_cpu_v4_fit_starts_before_eof_and_publishes_after_close(
    tmp_path,
    postgres_config,
    postgres_ledger,
):
    config = _service_config(tmp_path)
    application = FlightApplication.build(
        config,
        database_config=postgres_config,
        models_dir=tmp_path / "models",
        bearer_tokens={TOKEN: OWNER},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    try:
        job_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        created = _action(
            client,
            CREATE_ACTION,
            idempotencyKey=f"create-{job_id}",
            jobId=job_id,
            clientExecutionId=execution_id,
            operation="fit",
            device="cpu",
            modelLabel="integration-fit",
            modelConfig=MODEL_CONFIG,
            trainingConfig=TRAINING_CONFIG,
            dataContract=DATA_CONTRACT,
            mlContract=ML_CONTRACT,
        )
        assert created["jobId"] == job_id
        first = _put(client, created, 0, [
            _batch(
                [[0.10, 0.20, 0.30, 0.40]],
                [[0.05, 0.0, 0.0, 0.0, 0.20, 1.0]],
            ),
        ])
        assert first["queued"] is True

        deadline = time.monotonic() + 10
        while True:
            open_status = _action(client, STATUS_ACTION, jobId=job_id)
            if open_status["execution"]["state"] == ExecutionState.RUNNING.value:
                break
            assert time.monotonic() < deadline, open_status
            time.sleep(0.02)
        assert open_status["input"]["state"] == InputState.OPEN.value

        second = _put(client, created, 1, [
            _batch(
                [[0.90, 1.00, 1.10, 1.20]],
                [[0.02, 0.0, 0.0, 0.0, 0.30, 1.0]],
            ),
        ])
        receipts = application.ledger.list_inputs(job_id)
        assert [item["ordinal"] for item in receipts] == [0, 1]
        assert {item["storage_class"] for item in receipts} == {"recovery"}
        assert all(
            Path(application.recovery_store.absolute_path(
                item["relative_path"]
            )).is_file()
            for item in receipts
        )

        closed = _action(
            client,
            INPUT_CLOSE_ACTION,
            idempotencyKey=f"close-{job_id}",
            jobId=job_id,
            clientExecutionId=created["ownership"]["clientExecutionId"],
            fencingToken=created["ownership"]["fencingToken"],
            payloadCount=2,
            totalRows=first["rows"] + second["rows"],
            totalBytes=first["bytes"] + second["bytes"],
            manifestSha256=manifest_sha256(receipts),
        )
        assert closed["input"]["state"] == InputState.CLOSED.value

        deadline = time.monotonic() + 30
        while True:
            status = _action(client, STATUS_ACTION, jobId=job_id)
            if status["execution"]["state"] in {
                state.value
                for state in (
                    ExecutionState.SUCCEEDED,
                    ExecutionState.FAILED,
                    ExecutionState.CANCELLED,
                )
            }:
                break
            assert time.monotonic() < deadline, status
            time.sleep(0.02)

        assert status["execution"]["state"] == ExecutionState.SUCCEEDED.value, status
        assert status["input"]["state"] == InputState.CLOSED.value
        assert status["results"]["outputCount"] == 0
        assert status["results"]["modelRef"].startswith("mdl_")
        assert status["recovery"]["latestCheckpoint"]["completedEpochs"] == 1
        assert _action(client, CAPABILITIES_ACTION)["protocolVersions"] == [4]
        assert _action(client, HEALTH_ACTION)["live"] is True

        model = application.ledger.get_model(
            status["results"]["modelRef"],
            owner_subject=OWNER,
        )
        assert model is not None
        checkpoint_path = application.spool.model_absolute_path(
            model["checkpoint_path"]
        )
        assert Path(checkpoint_path).is_file()
        assert hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() == model["sha256"]
    finally:
        client.close()
        application.shutdown()
