from __future__ import annotations

import hashlib
import io
import json
import queue
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from app.contracts.worker.v2 import (
    encode_event,
    load_document,
    parse_control_message,
    parse_event,
    validate_document,
)
from app.contracts.worker.v2.config import (
    ModelConfig,
    TrainConfig,
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.service.adapters.outbound.worker_process.runner import (
    WorkerSubprocessRunner,
    _WorkerEventState,
)
from app.service.application.ports.workers import ExecutionInput
from app.service.domain.errors import failed_precondition
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import ExecutionJobRecord
from app.storage.checkpoint import save_checkpoint
from app.training.factory import build_model
from app.worker.application import executor as worker_executor

PROJECT_ROOT = Path(__file__).parents[1]
DATA_CONTRACT_SHA256 = "c" * 64
MANIFEST_SHA256 = "d" * 64


def _data_contract() -> dict:
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "seqLen": 2,
        "featureDim": 2,
        "targetSchemaId": "inventory.target.v1",
    }


def _write_input(path: Path, rows, *, fit: bool) -> None:
    fields = [pa.field("src", pa.list_(pa.float32(), 4), nullable=False)]
    arrays = [pa.array(rows, type=fields[0].type)]
    if fit:
        fields.append(pa.field("tgt", pa.list_(pa.float32(), 6), nullable=False))
        arrays.append(pa.array(
            [[0.0, 0.0, 0.0, 0.0, 0.2, 1.0] for _ in rows],
            type=fields[1].type,
        ))
    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)


def _input_manifest(path: Path, ordinal: int, rows: int, *, fit: bool) -> dict:
    return {
        "schemaId": (
            "inventory.sequence.fit.v2"
            if fit
            else "inventory.sequence.predict.v2"
        ),
        "ordinal": ordinal,
        "commitRevision": ordinal + 1,
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "rows": rows,
        "artifact": _artifact(path),
    }


def test_worker_capabilities_are_reported_through_v2_process_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "inspect",
            "--contract-version=2",
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
    assert document["protocolVersion"] == 2
    assert document["torchVersion"]


def test_worker_verifies_an_immutable_input_receipt_once(tmp_path, monkeypatch):
    input_path = tmp_path / "input.arrow"
    _write_input(input_path, [[1.0, 2.0, 3.0, 4.0]], fit=False)
    item = _input_manifest(input_path, 0, 1, fit=False)
    digest_calls = []
    sha256_file = worker_executor._sha256_file

    def counted_sha256(path):
        digest_calls.append(path)
        return sha256_file(path)

    monkeypatch.setattr(worker_executor, "_sha256_file", counted_sha256)
    committed = worker_executor._CommittedInputArtifacts()

    assert committed.path(item) == str(input_path)
    assert committed.path(item) == str(input_path)
    assert digest_calls == [str(input_path)]

    changed = dict(item)
    changed["commitRevision"] = 2
    with pytest.raises(ValueError, match="receipt changed"):
        committed.path(changed)


def test_worker_error_event_does_not_expose_manifest_diagnostics(tmp_path):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    marker = "secret-manifest-detail"
    manifest_path = tmp_path / "worker-command.json"
    manifest_path.write_text(json.dumps({"secret_marker": marker}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "run",
            "--contract-version=2",
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


def test_closed_predict_worker_publishes_only_one_terminal_result(tmp_path):
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
    _write_input(input_path, [[1.0, 2.0, 3.0, 4.0]], fit=False)

    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 2,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "predict",
        "predictionColumn": "predictions",
        "device": {"kind": "cpu"},
        "inputs": [_input_manifest(input_path, 0, 1, fit=False)],
        "inputRevision": 1,
        "inputClosed": True,
        "manifestSha256": MANIFEST_SHA256,
        "workspace": {"root": str(workspace)},
        "model": {
            "config": model_config_to_manifest(model_config),
            "checkpoint": _artifact(checkpoint),
        },
        "dataContract": _data_contract(),
    }
    manifest_path = workspace / "worker-command.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_worker(manifest_path, job_id, attempt_id)

    assert result.returncode == 0, result.stderr.decode()
    events = [parse_event(line) for line in result.stdout.splitlines(keepends=True)]
    assert [event["type"] for event in events] == ["ready", "progress", "completed"]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    result_manifest = load_document(
        workspace / "worker-result.json",
        "result-manifest",
    )
    assert result_manifest["inputRevision"] == 1
    assert result_manifest["manifestSha256"] == MANIFEST_SHA256
    assert result_manifest["artifacts"][0]["commitRevision"] == 1
    output = Path(result_manifest["artifacts"][0]["artifact"]["path"])
    with pa.memory_map(str(output), "r") as source_file:
        output_table = ipc.RecordBatchFileReader(source_file).read_all()
    assert output_table.schema.names == ["predictions"]
    assert output_table.num_rows == 1
    assert output_table.schema.field(0).type == pa.list_(pa.float32(), 6)


def test_closed_fit_worker_commits_global_epoch_checkpoint_and_result(tmp_path):
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
        _write_input(input_path, rows, fit=True)
        inputs.append(_input_manifest(input_path, ordinal, len(rows), fit=True))

    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 2,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "fit",
        "device": {"kind": "cpu"},
        "inputs": inputs,
        "inputRevision": 2,
        "inputClosed": True,
        "manifestSha256": MANIFEST_SHA256,
        "workspace": {"root": str(workspace)},
        "model": {
            "label": "returns.daily",
            "config": model_config_to_manifest(model_config),
        },
        "training": train_config_to_manifest(train_config),
        "dataContract": _data_contract(),
        "recovery": {
            "configSha256": "a" * 64,
            "dataContractSha256": DATA_CONTRACT_SHA256,
            "manifestSha256": MANIFEST_SHA256,
        },
    }
    manifest_path = workspace / "worker-command.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_worker(manifest_path, job_id, attempt_id, timeout=45)

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

    result_manifest = load_document(
        workspace / "worker-result.json",
        "result-manifest",
    )
    assert result_manifest["inputRevision"] == 2
    assert result_manifest["manifestSha256"] == MANIFEST_SHA256
    assert result_manifest["artifacts"] == []
    assert result_manifest["checkpointMetadata"]["dataContract"] == _data_contract()
    assert Path(result_manifest["checkpoint"]["path"]).is_file()


def test_service_rejects_progress_after_attempt_ownership_changes():
    attempt_id = str(uuid.uuid4())
    job = ExecutionJobRecord(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="predict",
        input_state=InputState.CLOSED,
        execution_state=ExecutionState.RUNNING,
        input_revision=1,
        selected_device="cpu",
        model_label=None,
        input_model_ref="mdl_seed",
        prediction_column="predictions",
        model_config=ModelConfig(seq_len=2, feature_dim=2),
        training_config=None,
        data_contract={"data_contract_sha256": DATA_CONTRACT_SHA256},
        config_hash="a" * 64,
        manifest_sha256=MANIFEST_SHA256,
        feature_dim=2,
        input_frame_count=0,
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
            return replace(job, attempt=2, attempt_id=str(uuid.uuid4()))

    runner = WorkerSubprocessRunner(
        object(),
        StaleLedger(),
        object(),
        logger=object(),
        python_executable=sys.executable,
    )
    stream = io.BytesIO(b"".join((
        encode_event(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=job.attempt_id,
            sequence=1,
            event_type="ready",
            payload={"pid": 1234, "nextOrdinal": 0, "inputRevision": 1},
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
        SimpleNamespace(inputs=()),
        _WorkerEventState(),
        errors,
    )

    failure = errors.get_nowait()
    assert failure.code == ErrorCode.EXECUTION_INTERRUPTED
    assert failure.message == "worker attempt no longer owns job progress"


def test_duplicate_worker_event_is_rejected_before_repeating_its_side_effect():
    attempt_id = str(uuid.uuid4())
    job = ExecutionJobRecord(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.OPEN,
        execution_state=ExecutionState.RUNNING,
        input_revision=1,
        selected_device="cpu",
        model_label="forecast",
        input_model_ref=None,
        prediction_column="out",
        model_config=ModelConfig(seq_len=2, feature_dim=2),
        training_config=TrainConfig(),
        data_contract={"data_contract_sha256": DATA_CONTRACT_SHA256},
        config_hash="a" * 64,
        manifest_sha256=None,
        feature_dim=2,
        input_frame_count=0,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
        attempt_id=attempt_id,
    )

    class Ledger:
        def __init__(self):
            self.waits = 0

        def mark_input_waiting(self, *args, **kwargs):
            self.waits += 1
            return True

    ledger = Ledger()
    runner = WorkerSubprocessRunner(
        object(),
        ledger,
        object(),
        logger=object(),
        python_executable=sys.executable,
    )
    waiting = encode_event(
        job_id=job.job_id,
        attempt=job.attempt,
        attempt_id=job.attempt_id,
        sequence=2,
        event_type="input.waiting",
        payload={"nextOrdinal": 0, "inputRevision": 1},
    )
    stream = io.BytesIO(b"".join((
        encode_event(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=job.attempt_id,
            sequence=1,
            event_type="ready",
            payload={"pid": 1234, "nextOrdinal": 0, "inputRevision": 1},
        ),
        waiting,
        waiting,
    )))
    errors = queue.Queue()

    runner._read_worker_events(
        stream,
        job,
        SimpleNamespace(inputs=()),
        _WorkerEventState(),
        errors,
    )

    failure = errors.get_nowait()
    assert ledger.waits == 1
    assert failure.code == ErrorCode.WORKER_PROTOCOL_VIOLATION
    assert failure.message == "worker event sequence is not contiguous"


def test_worker_control_poll_recovers_a_lost_input_notification():
    job = ExecutionJobRecord(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.OPEN,
        execution_state=ExecutionState.RUNNING,
        input_revision=0,
        selected_device="cpu",
        model_label="forecast",
        input_model_ref=None,
        prediction_column="out",
        model_config=ModelConfig(seq_len=2, feature_dim=2),
        training_config=TrainConfig(),
        data_contract={"data_contract_sha256": DATA_CONTRACT_SHA256},
        config_hash="a" * 64,
        manifest_sha256=None,
        feature_dim=2,
        input_frame_count=0,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
        attempt_id=str(uuid.uuid4()),
    )
    committed = ExecutionInput(
        ordinal=0,
        commit_revision=1,
        schema_id="inventory.sequence.fit.v2",
        data_contract_sha256=DATA_CONTRACT_SHA256,
        rows=2,
        byte_count=10,
        sha256="b" * 64,
        absolute_path="/srv/transformer/recovery/0.arrow",
        storage_class="recovery",
    )
    closed = replace(
        job,
        input_state=InputState.CLOSED,
        input_revision=1,
        input_frame_count=1,
        manifest_sha256=MANIFEST_SHA256,
    )

    class Ledger:
        def __init__(self):
            self.reads = 0

        def get_execution_job(self, job_id):
            assert job_id == job.job_id
            self.reads += 1
            return job if self.reads == 1 else closed

        def get_job(self, job_id):
            assert job_id == job.job_id
            return {
                "input_revision": 1,
                "payload_count": 1,
                "total_rows": 2,
                "total_bytes": 10,
                "manifest_sha256": MANIFEST_SHA256,
            }

    class PollTimeout:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            assert timeout == 0.25
            self.waits += 1
            return False

        def clear(self):
            pass

    class CaptureStream(io.BytesIO):
        def close(self):
            pass

    ledger = Ledger()
    wake = PollTimeout()
    stream = CaptureStream()
    errors = queue.Queue()
    runner = WorkerSubprocessRunner(
        object(),
        ledger,
        object(),
        logger=object(),
        python_executable=sys.executable,
        stream_inputs=lambda current, start: (
            (committed,)
            if current.input_state is InputState.CLOSED and start == 0
            else ()
        ),
    )

    runner._feed_worker_controls(
        stream,
        job,
        SimpleNamespace(inputs=()),
        wake,
        SimpleNamespace(is_set=lambda: False),
        errors,
    )

    messages = [
        parse_control_message(line)
        for line in stream.getvalue().splitlines(keepends=True)
    ]
    assert wake.waits == 1
    assert errors.empty()
    assert [message["type"] for message in messages] == [
        "input.committed",
        "input.closed",
    ]


def _run_worker(
    manifest_path: Path,
    job_id: str,
    attempt_id: str,
    *,
    timeout: int = 30,
):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "run",
            "--contract-version=2",
            f"--job-id={job_id}",
            "--attempt=1",
            f"--attempt-id={attempt_id}",
            f"--manifest={manifest_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        timeout=timeout,
    )


def _artifact(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "byteCount": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
