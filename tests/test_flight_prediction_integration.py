from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.flight as flight
import torch

from app.contracts.worker.v3.objective import (
    ml_contract,
    objective_config,
)
from app.flight.application import FlightApplication
from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CONTRACT_NAME,
    CREATE_ACTION,
    INPUT_CLOSE_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_SCHEMA_ID,
    STATUS_ACTION,
)
from app.flight.spool import Spool
from app.model.transformer import TransformerModel
from app.service.domain.input_manifest import manifest_sha256
from app.service.domain.job import ExecutionState, InputState
from app.storage.checkpoint import save_checkpoint
from tests.flight_v4_helpers import (
    DATA_CONTRACT_SHA256,
    OWNER,
    close_input,
    commit_input,
    create_fit,
    internal_data_contract,
    model_config,
    public_data_contract,
    public_ml_contract,
    train_config,
)

TOKEN = "secret"


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


def _publish_seed_model(ledger, models_dir, runtime_dir):
    spool = Spool(runtime_dir, models_dir).initialize()
    fit = create_fit(ledger)
    commit_input(ledger, fit, 0)
    close_input(ledger, fit)
    running = ledger.claim_execution_job(fit["job_id"], "cpu")
    assert running is not None

    config = model_config()
    source = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = TransformerModel(
        input_dim=2,
        seq_len=config.seq_len,
        hidden_dim=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
        out_dim=config.out_dim,
        nhead=config.nhead,
        context_mode=config.context_mode,
    )
    assert model(source).shape == (1, 7)
    model_ref = f"mdl_{uuid.uuid4().hex}"
    checkpoint = spool.model_checkpoint_path(model_ref)
    save_checkpoint(
        checkpoint,
        model,
        model_config=config,
        train_config=train_config(),
        data_contract=public_data_contract(),
    )
    metadata_path = spool.model_metadata_path(model_ref)
    metadata = {
        "model_config": config.to_dict(),
        "train_config": train_config().to_dict(),
        "data_contract": internal_data_contract(),
        "ml_contract": ml_contract(train_config()),
        "objective_config": objective_config(train_config()),
        "checkpoint": {"mlContract": public_ml_contract()},
    }
    spool.atomic_write_json(metadata_path, metadata)
    checkpoint_bytes = Path(checkpoint).read_bytes()
    ledger.publish_model(
        fit["job_id"],
        running.attempt,
        attempt_id=running.attempt_id,
        model_ref=model_ref,
        label="daily",
        generation=None,
        checkpoint_path=spool.model_relative_path(checkpoint),
        metadata_path=spool.model_relative_path(metadata_path),
        byte_count=len(checkpoint_bytes),
        sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        metadata=metadata,
        result={"modelRef": model_ref},
    )
    return model_ref


def _predict_schema():
    return pa.schema([
        pa.field("src", pa.list_(pa.float32(), 4), nullable=False),
    ])


def _put(client, created, ordinal, batches):
    payload_id = str(uuid.uuid4())
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v4", "jobs", created["jobId"], "inputs", str(ordinal)
    )
    writer, results = client.do_put(descriptor, _predict_schema(), options=_auth())
    rows = sum(batch.num_rows for batch in batches)
    writer.write_metadata(pa.py_buffer(json.dumps({
        "contract": CONTRACT_NAME,
        "version": 4,
        "jobId": created["jobId"],
        "clientExecutionId": created["ownership"]["clientExecutionId"],
        "fencingToken": created["ownership"]["fencingToken"],
        "payloadId": payload_id,
        "ordinal": ordinal,
        "schemaId": PREDICT_SCHEMA_ID,
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


def _create_predict(client, model_ref):
    job_id = str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    return _action(
        client,
        CREATE_ACTION,
        idempotencyKey=f"create-{job_id}",
        jobId=job_id,
        clientExecutionId=execution_id,
        operation="predict",
        device="cpu",
        modelRef=model_ref,
        predictionColumn="prediction",
        dataContract=public_data_contract(),
        mlContract=public_ml_contract(),
    )


def _close(client, application, created):
    receipts = application.ledger.list_inputs(created["jobId"])
    return _action(
        client,
        INPUT_CLOSE_ACTION,
        idempotencyKey=f"close-{created['jobId']}",
        jobId=created["jobId"],
        clientExecutionId=created["ownership"]["clientExecutionId"],
        fencingToken=created["ownership"]["fencingToken"],
        payloadCount=len(receipts),
        totalRows=sum(item["rows"] for item in receipts),
        totalBytes=sum(item["bytes"] for item in receipts),
        manifestSha256=manifest_sha256(receipts),
    )


def _wait_terminal(client, job_id):
    deadline = time.monotonic() + 30
    while True:
        status = _action(client, STATUS_ACTION, jobId=job_id)
        if status["execution"]["state"] in {
            ExecutionState.SUCCEEDED.value,
            ExecutionState.FAILED.value,
            ExecutionState.CANCELLED.value,
        }:
            return status
        assert time.monotonic() < deadline, status
        time.sleep(0.02)


def test_real_cpu_v4_prediction_publishes_outputs_atomically_after_eof(
    tmp_path,
    postgres_ledger,
    postgres_config,
):
    config = _service_config(tmp_path)
    models_dir = tmp_path / "models"
    model_ref = _publish_seed_model(
        postgres_ledger,
        models_dir,
        config.runtime_dir,
    )
    application = FlightApplication.build(
        config,
        database_config=postgres_config,
        models_dir=models_dir,
        bearer_tokens={TOKEN: OWNER},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    try:
        created = _create_predict(client, model_ref)
        schema = _predict_schema()
        nonempty = pa.RecordBatch.from_arrays(
            [pa.array(
                [[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]],
                type=schema.field("src").type,
            )],
            schema=schema,
        )
        first = _put(client, created, 0, [nonempty])
        second = _put(client, created, 1, [])
        assert (first["rows"], second["rows"]) == (2, 0)

        closed = _close(client, application, created)
        assert closed["input"]["state"] == InputState.CLOSED.value
        status = _wait_terminal(client, created["jobId"])
        assert status["execution"]["state"] == ExecutionState.SUCCEEDED.value, status
        assert status["results"]["outputCount"] == 2

        outputs = _action(
            client,
            OUTPUTS_LIST_ACTION,
            jobId=created["jobId"],
            limit=100,
        )
        assert [(item["ordinal"], item["rows"]) for item in outputs["items"]] == [
            (0, 2),
            (1, 0),
        ]
        for ordinal, expected_rows in ((0, 2), (1, 0)):
            descriptor = flight.FlightDescriptor.for_path(
                "transformer", "v4", "jobs", created["jobId"],
                "outputs", str(ordinal),
            )
            info = client.get_flight_info(descriptor, options=_auth())
            output = client.do_get(info.endpoints[0].ticket, options=_auth()).read_all()
            assert output.num_rows == expected_rows
            assert output.schema == pa.schema([
                pa.field(
                    "prediction",
                    pa.list_(pa.float32(), 6),
                    nullable=False,
                ),
            ])
    finally:
        client.close()
        application.shutdown()


def test_empty_predict_closes_and_succeeds_with_no_outputs(
    tmp_path,
    postgres_ledger,
    postgres_config,
):
    config = _service_config(tmp_path)
    models_dir = tmp_path / "models"
    model_ref = _publish_seed_model(
        postgres_ledger,
        models_dir,
        config.runtime_dir,
    )
    application = FlightApplication.build(
        config,
        database_config=postgres_config,
        models_dir=models_dir,
        bearer_tokens={TOKEN: OWNER},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    try:
        created = _create_predict(client, model_ref)
        _close(client, application, created)
        status = _wait_terminal(client, created["jobId"])
        assert status["execution"]["state"] == ExecutionState.SUCCEEDED.value, status
        assert status["input"]["payloadCount"] == 0
        assert status["results"]["outputCount"] == 0
    finally:
        client.close()
        application.shutdown()
