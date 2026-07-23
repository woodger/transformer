from dataclasses import replace
from io import BytesIO
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import torch

from app.config import PROJECT_ROOT
from app.data.arrow import iter_framed_arrow
from app.flight.application import FlightApplication
from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CONTRACT_NAME,
    CREATE_ACTION,
    JobState,
    PREDICT_SCHEMA_ID,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
)
from app.flight.ledger import Ledger
from app.flight.spool import Spool
from app.model.transformer import TransformerModel
from app.runtime.version import __version__
from app.storage.checkpoint import CHECKPOINT_FORMAT
from app.training.run_config import ModelConfig, TrainConfig


OWNER = "inventory"
TOKEN = "secret"
PREDICTION_COLUMN = "prediction"


def _service_config(tmp_path):
    return FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        allow_plaintext=True,
        disk_min_free_bytes=1,
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


def _request(**fields):
    return {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": str(uuid.uuid4()),
        **fields,
    }


def _action(client, name, **fields):
    document = _request(**fields)
    results = list(client.do_action(
        flight.Action(name, json.dumps(document).encode()),
        options=_auth(),
    ))
    assert len(results) == 1
    return json.loads(results[0].body.to_pybytes())


def _predict_schema():
    return pa.schema([("src", pa.list_(pa.float32()))])


def _batch(rows):
    return pa.record_batch(
        [pa.array(rows, type=pa.list_(pa.float32()))],
        schema=_predict_schema(),
    )


def _ipc_payload(schema, batches):
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _framed_payloads(payloads):
    stream = BytesIO()
    for payload in payloads:
        stream.write(len(payload).to_bytes(8, "big"))
        stream.write(payload)
    stream.write((0).to_bytes(8, "big"))
    return stream.getvalue()


def _put(client, job_id, payload_id, ordinal, batches):
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job_id, "inputs", str(ordinal)
    )
    writer, results = client.do_put(
        descriptor,
        _predict_schema(),
        options=_auth(),
    )
    metadata = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "jobId": job_id,
        "payloadId": payload_id,
        "ordinal": ordinal,
        "schemaId": PREDICT_SCHEMA_ID,
        "rows": sum(batch.num_rows for batch in batches),
    }
    writer.write_metadata(pa.py_buffer(json.dumps(metadata).encode()))
    for batch in batches:
        writer.write_batch(batch)
    writer.done_writing()
    result = results.read()
    assert results.read() is None
    writer.close()
    return json.loads(result.to_pybytes())


def _streamed_table(client, ticket):
    reader = client.do_get(ticket, options=_auth())
    schema = reader.schema
    batches = [chunk.data for chunk in reader if chunk.data is not None]
    return pa.Table.from_batches(batches, schema=schema), len(batches)


def _seed_model(config, ledger, models_dir):
    spool = Spool(config.runtime_dir, models_dir).initialize()
    model_config = ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        context_mode="relaxed",
        feature_dim=2,
    )
    train_config = TrainConfig(
        epochs=1,
        patience=0,
        loss_stage=1,
        loss_schedule="none",
        stage_size=1,
        save_best_checkpoint=False,
        seed=17,
        deterministic=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = TransformerModel(
            input_dim=model_config.feature_dim,
            seq_len=model_config.seq_len,
            hidden_dim=model_config.hidden,
            layers=model_config.layers,
            dropout=model_config.dropout,
            out_dim=model_config.out_dim,
            nhead=model_config.nhead,
            context_mode=model_config.context_mode,
        )
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "version": __version__,
        "state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "data_schema": {
            "schema_version": 1,
            "tensor_dtype": "float32",
            "src": {
                "column": "src",
                "accepted_element_types": ["float32", "float64"],
                "width": 4,
            },
            "tgt": {
                "column": "tgt",
                "accepted_element_types": ["float32", "float64"],
                "width": 6,
            },
            "feature_dim": 2,
            "model_input_feature_dim": 4,
            "context_mode": "relaxed",
            "normalization": None,
            "missing": {"nan_fill": 0.0, "flags": "per-feature"},
        },
    }
    checkpoint_buffer = BytesIO()
    torch.save(checkpoint, checkpoint_buffer)

    producer_id = str(uuid.uuid4())
    producer = ledger.create_job(
        job_id=producer_id,
        owner_subject=OWNER,
        operation="fit",
        requested_device="cpu",
        prediction_column=PREDICTION_COLUMN,
        config_hash="a" * 64,
        model_label="integration-seed",
        model_config=replace(model_config, feature_dim=None),
        training_config=train_config,
    )
    ledger.seal_job(
        producer_id,
        manifest_hash=hashlib.sha256(b"[]").hexdigest(),
        manifest=[],
    )
    ledger.queue_job(producer_id, selected_device="cpu")
    running = ledger.claim_next_job("cpu", worker_id="integration-seed")
    assert running is not None

    model_ref = f"mdl_integration_{uuid.uuid4().hex}"
    checkpoint_path = spool.model_checkpoint_path(model_ref)
    metadata_path = spool.model_metadata_path(model_ref)
    checkpoint_bytes = checkpoint_buffer.getvalue()
    metadata = {
        "modelRef": model_ref,
        "label": "integration-seed",
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "checkpoint": {},
    }
    spool.atomic_write_bytes(checkpoint_path, checkpoint_bytes)
    spool.atomic_write_json(metadata_path, metadata)
    ledger.publish_model(
        producer["job_id"],
        running["attempt"],
        model_ref=model_ref,
        label="integration-seed",
        generation=None,
        checkpoint_path=spool.model_relative_path(checkpoint_path),
        metadata_path=spool.model_relative_path(metadata_path),
        sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        metadata=metadata,
        result={"modelRef": model_ref, "checkpoint": {}},
    )
    return model_ref, checkpoint_path


def _direct_predict(checkpoint_path, payloads):
    process = subprocess.run(
        [
            sys.executable,
            str(Path(PROJECT_ROOT) / "app" / "main.py"),
            "predict-stream",
            "--device", "cpu",
            "--checkpoint", checkpoint_path,
            "--pred-col", PREDICTION_COLUMN,
        ],
        cwd=PROJECT_ROOT,
        input=_framed_payloads(payloads),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    return list(iter_framed_arrow(BytesIO(process.stdout)))


def test_real_cpu_flight_prediction_matches_predict_stream(
    tmp_path,
    postgres_ledger,
    postgres_config,
):
    config = _service_config(tmp_path)
    models_dir = tmp_path / "models"
    model_ref, checkpoint_path = _seed_model(
        config,
        postgres_ledger,
        models_dir,
    )
    first_batches = [
        _batch([
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 1.1, 1.2, 1.3],
        ]),
        _batch([[2.0, 2.1, 2.2, 2.3]]),
    ]
    empty_batches = []
    input_payloads = [
        _ipc_payload(_predict_schema(), first_batches),
        _ipc_payload(_predict_schema(), empty_batches),
    ]
    expected = _direct_predict(checkpoint_path, input_payloads)
    assert [table.num_rows for table in expected] == [3, 0]

    application = FlightApplication.build(
        config,
        database_config=postgres_config,
        models_dir=models_dir,
        bearer_tokens={TOKEN: OWNER},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    try:
        created = _action(
            client,
            CREATE_ACTION,
            idempotencyKey=f"create-{uuid.uuid4()}",
            operation="predict",
            device="cpu",
            modelRef=model_ref,
            predictionColumn=PREDICTION_COLUMN,
        )
        job_id = created["jobId"]
        payload_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        committed = [
            _put(client, job_id, payload_ids[0], 0, first_batches),
            _put(client, job_id, payload_ids[1], 1, empty_batches),
        ]
        assert [(item["ordinal"], item["rows"], item["batches"]) for item in committed] == [
            (0, 3, 2),
            (1, 0, 0),
        ]

        manifest = [
            {
                "payloadId": payload_ids[index],
                "ordinal": index,
                "sha256": committed[index]["sha256"],
            }
            for index in range(2)
        ]
        sealed = _action(
            client,
            SEAL_ACTION,
            idempotencyKey=f"seal-{uuid.uuid4()}",
            jobId=job_id,
            manifest=manifest,
        )
        assert sealed["state"] == JobState.SEALED.value
        started = _action(
            client,
            START_ACTION,
            idempotencyKey=f"start-{uuid.uuid4()}",
            jobId=job_id,
        )
        assert started["state"] == JobState.QUEUED.value

        deadline = time.monotonic() + 30
        while True:
            status = _action(client, STATUS_ACTION, jobId=job_id)
            if status["state"] in {
                JobState.SUCCEEDED.value,
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            }:
                break
            assert time.monotonic() < deadline, status
            time.sleep(0.05)
        assert status["state"] == JobState.SUCCEEDED.value, status
        assert [
            (item["ordinal"], item["rows"], item["batches"])
            for item in status["committedInputs"]
        ] == [(0, 3, 2), (1, 0, 0)]
        assert [
            (item["ordinal"], item["rows"])
            for item in status["results"]["outputs"]
        ] == [(0, 3), (1, 0)]

        actual = []
        output_batch_counts = []
        for ordinal in range(2):
            descriptor = flight.FlightDescriptor.for_path(
                "transformer", "v1", "jobs", job_id, "outputs", str(ordinal)
            )
            info = client.get_flight_info(descriptor, options=_auth())
            assert info.descriptor == descriptor
            assert info.total_records == (3 if ordinal == 0 else 0)
            assert len(info.endpoints) == 1
            table, batch_count = _streamed_table(client, info.endpoints[0].ticket)
            actual.append(table)
            output_batch_counts.append(batch_count)

        assert actual[0].equals(expected[0])
        assert actual[1].equals(expected[1])
        assert actual[0].schema == pa.schema([
            (PREDICTION_COLUMN, pa.list_(pa.float32()))
        ])
        assert all(len(row) == 6 for row in actual[0][PREDICTION_COLUMN].to_pylist())
        assert actual[1].num_rows == 0
        assert actual[1].schema == actual[0].schema
        assert output_batch_counts[1] == 0

        stderr_path = application.spool.attempt_stderr_path(job_id, status["attempt"])
        stderr = Path(stderr_path).read_text(errors="replace")
        assert stderr.count("Model loaded") == 1
        assert "from 2 received frame(s)" in stderr
    finally:
        client.close()
        application.shutdown()
