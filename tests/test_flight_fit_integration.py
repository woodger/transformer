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
from app.flight.application import FlightApplication
from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    JobState,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
)
from app.runtime.version import __version__
from app.storage.checkpoint import CHECKPOINT_FORMAT


OWNER = "inventory"
TOKEN = "secret"
MODEL_LABEL = "integration-fit"

MODEL_CONFIG = {
    "seqLen": 2,
    "hidden": 4,
    "layers": 1,
    "dropout": 0.0,
    "nhead": 2,
    "mode": "relaxed",
}
TRAINING_CONFIG = {
    "lr": 0.001,
    "weightDecay": 0.0,
    "batchSize": 8,
    "epochs": 1,
    "patience": 0,
    "lossStage": 1,
    "lossSchedule": "none",
    "stageSize": 1,
    "useAmp": False,
    "monitor": "loss",
    "monitorMinImprovement": 0.0,
    "saveBestCheckpoint": False,
    "seed": 29,
    "deterministic": True,
}


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


def _action(client, name, **fields):
    request = {
        "contract": CONTRACT_NAME,
        "version": 1,
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
        ("src", pa.list_(pa.float32())),
        ("tgt", pa.list_(pa.float32())),
    ])


def _batch(source_rows, target_rows):
    return pa.record_batch(
        [
            pa.array(source_rows, type=pa.list_(pa.float32())),
            pa.array(target_rows, type=pa.list_(pa.float32())),
        ],
        schema=_fit_schema(),
    )


def _ipc_payload(batches):
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, _fit_schema()) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _put(client, job_id, payload_id, ordinal, batches):
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job_id, "inputs", str(ordinal)
    )
    writer, results = client.do_put(
        descriptor,
        _fit_schema(),
        options=_auth(),
    )
    metadata = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "jobId": job_id,
        "payloadId": payload_id,
        "ordinal": ordinal,
        "schemaId": FIT_SCHEMA_ID,
        "rows": sum(batch.num_rows for batch in batches),
    }
    writer.write_metadata(pa.py_buffer(json.dumps(metadata).encode()))
    for batch in batches:
        writer.write_batch(batch)
    writer.done_writing()
    committed = results.read()
    assert committed is not None
    assert results.read() is None
    writer.close()
    return json.loads(committed.to_pybytes())


def _direct_fit(tmp_path, payloads):
    checkpoint = tmp_path / "direct-checkpoint.pth"
    metrics = tmp_path / "direct-metrics.jsonl"
    inputs = tmp_path / "direct-inputs"
    inputs.mkdir()
    for ordinal, payload in enumerate(payloads):
        (inputs / f"{ordinal}.arrow").write_bytes(payload)
    command = [
        sys.executable,
        str(Path(PROJECT_ROOT) / "app" / "main.py"),
        "fit-stream",
        "--device", "cpu",
        "--checkpoint-out", str(checkpoint),
        "--metrics-out", str(metrics),
        "--input-spool-dir", str(inputs),
        "--input-frame-count", str(len(payloads)),
        "--seq-len", str(MODEL_CONFIG["seqLen"]),
        "--hidden", str(MODEL_CONFIG["hidden"]),
        "--layers", str(MODEL_CONFIG["layers"]),
        "--dropout", str(MODEL_CONFIG["dropout"]),
        "--nhead", str(MODEL_CONFIG["nhead"]),
        "--mode", MODEL_CONFIG["mode"],
        "--lr", str(TRAINING_CONFIG["lr"]),
        "--weight-decay", str(TRAINING_CONFIG["weightDecay"]),
        "--batch-size", str(TRAINING_CONFIG["batchSize"]),
        "--epochs", str(TRAINING_CONFIG["epochs"]),
        "--loss-stage", str(TRAINING_CONFIG["lossStage"]),
        "--loss-schedule", TRAINING_CONFIG["lossSchedule"],
        "--stage-size", str(TRAINING_CONFIG["stageSize"]),
        "--patience", str(TRAINING_CONFIG["patience"]),
        "--monitor", TRAINING_CONFIG["monitor"],
        "--monitor-min-improvement",
        str(TRAINING_CONFIG["monitorMinImprovement"]),
        "--seed", str(TRAINING_CONFIG["seed"]),
        "--no-save-best-checkpoint",
        "--deterministic",
    ]
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        input=b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    return checkpoint, metrics, process.stdout.decode(errors="replace")


def _jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _without_elapsed(metrics):
    return [
        {key: value for key, value in row.items() if key != "elapsed_ms"}
        for row in metrics
    ]


def _load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_real_cpu_flight_fit_runs_global_epochs_over_spooled_payloads(
    tmp_path,
    postgres_config,
    postgres_database,
):
    first_batches = [
        _batch(
            [[0.10, 0.20, 0.30, 0.40]],
            [[0.05, 0.0, 0.0, 0.0, 0.20, 1.0]],
        ),
        _batch(
            [[0.50, 0.60, 0.70, 0.80]],
            [[-0.03, 0.0, 0.0, 0.0, 0.25, 0.0]],
        ),
    ]
    second_batches = [
        _batch(
            [[0.90, 1.00, 1.10, 1.20]],
            [[0.02, 0.0, 0.0, 0.0, 0.30, 1.0]],
        )
    ]
    payloads = [
        _ipc_payload(first_batches),
        _ipc_payload(second_batches),
    ]
    direct_checkpoint, direct_metrics_path, direct_stdout = _direct_fit(
        tmp_path,
        payloads,
    )
    assert "2 trained frame(s), 1 epoch(s) from 2 received frame(s)" in direct_stdout

    config = _service_config(tmp_path)
    application = FlightApplication.build(
        config,
        database_config=postgres_config,
        models_dir=tmp_path / "models",
        bearer_tokens={TOKEN: OWNER},
    )
    client = flight.FlightClient(("localhost", application.server.port))
    job_id = None
    try:
        created = _action(
            client,
            CREATE_ACTION,
            idempotencyKey=f"create-{uuid.uuid4()}",
            operation="fit",
            device="cpu",
            modelLabel=MODEL_LABEL,
            modelConfig=MODEL_CONFIG,
            trainingConfig=TRAINING_CONFIG,
        )
        job_id = created["jobId"]
        payload_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        committed = [
            _put(client, job_id, payload_ids[0], 0, first_batches),
            _put(client, job_id, payload_ids[1], 1, second_batches),
        ]
        assert [
            (item["ordinal"], item["rows"], item["batches"])
            for item in committed
        ] == [(0, 2, 2), (1, 1, 1)]

        manifest = [
            {
                "payloadId": payload_ids[ordinal],
                "ordinal": ordinal,
                "sha256": committed[ordinal]["sha256"],
            }
            for ordinal in range(2)
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

        saw_running = False
        deadline = time.monotonic() + 30
        while True:
            status = _action(client, STATUS_ACTION, jobId=job_id)
            if status["state"] == JobState.RUNNING.value and not saw_running:
                saw_running = True
                capabilities = _action(client, CAPABILITIES_ACTION)
                health = _action(client, HEALTH_ACTION)
                assert capabilities["supportedOperations"] == ["fit", "predict"]
                assert health["live"] is True
            if status["state"] in {
                JobState.SUCCEEDED.value,
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            }:
                break
            assert time.monotonic() < deadline, status
            time.sleep(0.02)

        assert saw_running
        assert status["state"] == JobState.SUCCEEDED.value, status
        assert [
            (item["ordinal"], item["rows"], item["batches"])
            for item in status["committedInputs"]
        ] == [(0, 2, 2), (1, 1, 1)]
        assert status["results"]["outputs"] == []

        model_ref = status["results"]["modelRef"]
        assert model_ref.startswith("mdl_")
        assert "/" not in model_ref and "\\" not in model_ref
        serialized_status = json.dumps(status)
        assert str(tmp_path) not in serialized_status
        assert ".pth" not in serialized_status

        checkpoint = status["results"]["checkpoint"]
        assert set(checkpoint) == {
            "format",
            "serviceVersion",
            "sha256",
            "bytes",
            "modelConfig",
            "trainConfig",
            "dataSchema",
            "checkpointSelection",
        }
        assert checkpoint["format"] == CHECKPOINT_FORMAT
        assert checkpoint["serviceVersion"] == __version__
        assert len(checkpoint["sha256"]) == 64
        assert checkpoint["bytes"] > 0
        assert checkpoint["modelConfig"] == {
            **MODEL_CONFIG,
            "outDim": 6,
            "featureDim": 2,
        }
        assert checkpoint["trainConfig"] == TRAINING_CONFIG
        assert checkpoint["dataSchema"] == {
            "schemaVersion": 1,
            "tensorDtype": "float32",
            "source": {
                "column": "src",
                "acceptedElementTypes": ["float32", "float64"],
                "width": 4,
            },
            "target": {
                "column": "tgt",
                "acceptedElementTypes": ["float32", "float64"],
                "width": 6,
            },
            "featureDim": 2,
            "modelInputFeatureDim": 4,
            "contextMode": "relaxed",
            "normalization": None,
            "missing": {"nanFill": 0.0, "flags": "per-feature"},
        }
        assert checkpoint["checkpointSelection"] == {
            "monitor": "loss",
            "monitorMinImprovement": 0.0,
            "bestMonitor": None,
            "bestFrame": None,
            "bestEpoch": None,
            "baselinePassed": False,
            "source": "current",
        }

        model = application.ledger.get_model(model_ref, owner_subject=OWNER)
        assert model is not None
        published_checkpoint = application.spool.model_absolute_path(
            model["checkpoint_path"]
        )
        assert checkpoint["bytes"] == Path(published_checkpoint).stat().st_size
        assert checkpoint["sha256"] == model["sha256"]

        direct_payload = _load_payload(direct_checkpoint)
        published_payload = _load_payload(published_checkpoint)
        assert direct_payload["model_config"] == published_payload["model_config"]
        assert direct_payload["train_config"] == published_payload["train_config"]
        assert direct_payload["state_dict"].keys() == published_payload["state_dict"].keys()
        for key, direct_value in direct_payload["state_dict"].items():
            assert torch.equal(direct_value, published_payload["state_dict"][key]), key

        service_metrics_path = application.spool.attempt_metrics_path(
            job_id,
            status["attempt"],
        )
        direct_metrics = _jsonl(direct_metrics_path)
        service_metrics = _jsonl(service_metrics_path)
        assert [row.get("frame") for row in service_metrics] == [None]
        assert [row["epoch"] for row in service_metrics] == [1]
        assert [row["rows"] for row in service_metrics] == [3]
        assert [row["batches"] for row in service_metrics] == [2]
        assert [row["step"] for row in service_metrics] == [2]
        assert _without_elapsed(service_metrics) == _without_elapsed(direct_metrics)

        service_stdout = Path(application.spool.attempt_stdout_path(
            job_id,
            status["attempt"],
        )).read_text(errors="replace")
        assert "2 trained frame(s), 1 epoch(s) from 2 received frame(s)" in service_stdout
    finally:
        client.close()
        application.shutdown()
