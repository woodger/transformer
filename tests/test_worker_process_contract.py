import hashlib
import io
import json
import queue
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from app.contracts.worker.v1 import (
    encode_event,
    load_document,
    parse_event,
    validate_document,
)
from app.contracts.worker.v1.config import (
    ModelConfig,
    TrainConfig,
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.service.adapters.outbound.worker_process.runner import (
    WorkerSubprocessRunner,
    _WorkerEventState,
)
from app.service.domain.errors import failed_precondition
from app.service.domain.job import ErrorCode, JobState
from app.service.domain.records import ExecutionJobRecord
from app.storage.checkpoint import save_checkpoint
from app.training.factory import build_model

PROJECT_ROOT = Path(__file__).parents[1]


def test_worker_capabilities_are_reported_through_versioned_process_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "inspect",
            "--contract-version=1",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    document = validate_document(json.loads(result.stdout), "capabilities")
    assert document["contract"] == "transformer-worker"
    assert document["protocolVersion"] == 1
    assert document["torchVersion"]


def test_worker_error_event_does_not_expose_manifest_diagnostics(tmp_path):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    marker = "secret-manifest-detail"
    manifest_path = tmp_path / "worker-command.json"
    manifest_path.write_text(
        json.dumps({"secret_marker": marker}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "run",
            "--contract-version=1",
            f"--job-id={job_id}",
            "--attempt=1",
            f"--attempt-id={attempt_id}",
            f"--manifest={manifest_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 1
    events = [parse_event(line) for line in result.stdout.splitlines(keepends=True)]
    assert [(event["type"], event["payload"]) for event in events] == [
        (
            "error",
            {
                "code": "WORKER_PROTOCOL_VIOLATION",
                "message": "worker command manifest is invalid",
            },
        )
    ]
    assert marker.encode() not in result.stdout


def test_predict_worker_uses_manifest_events_and_attempt_workspace(tmp_path):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    model_config = ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        feature_dim=2,
    )
    train_config = TrainConfig(
        batch_size=2,
        epochs=1,
        patience=0,
        loss_stage=1,
        loss_schedule="none",
        stage_size=1,
    )
    source = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = build_model(model_config, source, None, torch.device("cpu"))
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(
        str(checkpoint),
        model,
        model_config=model_config,
        train_config=train_config,
    )
    input_path = tmp_path / "input.arrow"
    table = pa.table({
        "src": pa.array(
            [[1.0, 2.0, 3.0, 4.0]],
            type=pa.list_(pa.float32()),
        )
    })
    with pa.OSFile(str(input_path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 1,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "predict",
        "predictionColumn": "predictions",
        "device": {"kind": "cpu"},
        "inputs": [{
            "schemaId": "inventory.sequence.predict.v1",
            "ordinal": 0,
            "rows": 1,
            "artifact": _artifact(input_path),
        }],
        "workspace": {"root": str(workspace)},
        "model": {
            "config": model_config_to_manifest(model_config),
            "checkpoint": _artifact(checkpoint),
        },
    }
    manifest_path = workspace / "worker-command.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "run",
            "--contract-version=1",
            f"--job-id={job_id}",
            "--attempt=1",
            f"--attempt-id={attempt_id}",
            f"--manifest={manifest_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr.decode()
    events = [
        parse_event(line)
        for line in result.stdout.splitlines(keepends=True)
    ]
    assert [event["type"] for event in events] == [
        "ready",
        "progress",
        "completed",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert all(event["attemptId"] == attempt_id for event in events)
    result_manifest = load_document(
        workspace / "worker-result.json",
        "result-manifest",
    )
    assert result_manifest["attemptId"] == attempt_id
    assert (
        result_manifest["artifacts"][0]["schemaId"]
        == "transformer.prediction.v1"
    )
    assert result_manifest["artifacts"][0]["rows"] == 1
    output = Path(result_manifest["artifacts"][0]["artifact"]["path"])
    with pa.memory_map(str(output), "r") as source_file:
        output_table = ipc.RecordBatchFileReader(source_file).read_all()
    assert output_table.schema.names == ["predictions"]
    assert output_table.num_rows == 1


def test_fit_worker_commits_global_epoch_checkpoint_and_result(tmp_path):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    model_config = ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
    )
    train_config = TrainConfig(
        batch_size=2,
        epochs=1,
        patience=0,
        loss_stage=1,
        loss_schedule="none",
        stage_size=1,
        monitor="loss",
        save_best_checkpoint=False,
        seed=7,
        deterministic=True,
    )
    inputs = []
    for ordinal, rows in enumerate((
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]],
    )):
        input_path = tmp_path / f"input-{ordinal}.arrow"
        table = pa.table({
            "src": pa.array(rows, type=pa.list_(pa.float32())),
            "tgt": pa.array(
                [[0.0, 0.0, 0.0, 0.0, 0.2, 1.0] for _ in rows],
                type=pa.list_(pa.float32()),
            ),
        })
        with pa.OSFile(str(input_path), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        inputs.append({
            "schemaId": "inventory.sequence.fit.v1",
            "ordinal": ordinal,
            "rows": len(rows),
            "artifact": _artifact(input_path),
        })

    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 1,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "fit",
        "device": {"kind": "cpu"},
        "inputs": inputs,
        "workspace": {"root": str(workspace)},
        "model": {
            "label": "returns.daily",
            "config": model_config_to_manifest(model_config),
        },
        "training": train_config_to_manifest(train_config),
        "recovery": {
            "configSha256": "a" * 64,
            "manifestSha256": "b" * 64,
        },
    }
    manifest_path = workspace / "worker-command.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "run",
            "--contract-version=1",
            f"--job-id={job_id}",
            "--attempt=1",
            f"--attempt-id={attempt_id}",
            f"--manifest={manifest_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr.decode()
    events = [parse_event(line) for line in result.stdout.splitlines(keepends=True)]
    assert [event["type"] for event in events] == [
        "ready",
        "progress",
        "checkpoint",
        "completed",
    ]
    checkpoint_event = events[2]["payload"]
    assert checkpoint_event["generation"] == 1
    assert checkpoint_event["completedEpochs"] == 1
    assert checkpoint_event["trainingComplete"] is True
    assert Path(checkpoint_event["artifact"]["path"]).is_file()

    result_manifest = load_document(
        workspace / "worker-result.json",
        "result-manifest",
    )
    assert result_manifest["attemptId"] == attempt_id
    assert result_manifest["operation"] == "fit"
    assert result_manifest["artifacts"] == []
    assert Path(result_manifest["checkpoint"]["path"]).is_file()
    assert Path(result_manifest["metrics"]["path"]).is_file()


def test_service_rejects_progress_after_attempt_ownership_changes():
    attempt_id = str(uuid.uuid4())
    job = ExecutionJobRecord(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="predict",
        state=JobState.RUNNING,
        selected_device="cpu",
        model_label=None,
        input_model_ref="mdl_seed",
        prediction_column="predictions",
        model_config=None,
        training_config=None,
        config_hash="a" * 64,
        seal_hash="b" * 64,
        feature_dim=2,
        input_frame_count=1,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
        attempt_id=attempt_id,
    )

    class StaleLedger:
        def update_progress(self, job_id, progress, *, attempt_id):
            assert (job_id, progress, attempt_id) == (
                job.job_id,
                {"ordinal": 0},
                job.attempt_id,
            )
            raise failed_precondition("job attempt is no longer active")

        def get_execution_job(self, job_id):
            assert job_id == job.job_id
            return replace(
                job,
                attempt=2,
                attempt_id=str(uuid.uuid4()),
            )

    runner = WorkerSubprocessRunner(
        object(),
        StaleLedger(),
        object(),
        logger=object(),
        python_executable=sys.executable,
        stage_prediction=lambda *_args: None,
    )
    stream = io.BytesIO(b"".join((
        encode_event(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=job.attempt_id,
            sequence=1,
            event_type="ready",
            payload={"pid": 1234},
        ),
        encode_event(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=job.attempt_id,
            sequence=2,
            event_type="progress",
            payload={"progress": {"ordinal": 0}},
        ),
    )))
    errors = queue.Queue()

    runner._read_worker_events(
        stream,
        job,
        object(),
        _WorkerEventState(),
        errors,
    )

    failure = errors.get_nowait()
    assert failure.code == ErrorCode.EXECUTION_INTERRUPTED
    assert failure.message == "worker attempt no longer owns job progress"


def _artifact(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "byteCount": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
