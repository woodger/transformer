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

from app.contracts.flight.v11.arrow import canonical_input_schema
from app.contracts.worker.v12 import (
    WorkerContractError,
    encode_event,
    load_document,
    parse_control_message,
    parse_event,
    validate_document,
)
from app.contracts.worker.v12.config import (
    ModelConfig,
    TrainConfig,
    train_config_to_manifest,
)
from app.project import PROJECT_ROOT
from app.service.adapters.outbound.worker.runner import (
    WorkerSubprocessError,
    WorkerSubprocessRunner,
    _WorkerEventState,
)
from app.service.application.ports.workers import ExecutionInput
from app.service.domain.errors import failed_precondition
from app.service.domain.initialization import published_model_initialization
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import ExecutionJobRecord
from app.worker.application import artifacts as worker_artifacts
from app.worker.checkpoints.model import load_checkpoint, save_checkpoint
from app.worker.training.factory import build_model
from tests.support.consumer_neutral import model_contract

DATA_CONTRACT_SHA256 = "c" * 64
MANIFEST_SHA256 = "d" * 64
SOURCE_ENCODING = {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
        {"position": 0, "windowRows": 1, "nativeRowWidth": 2},
    ],
}
MODEL_CONTRACT_VALUE = model_contract(
    "single-regression",
    seq_len=2,
    feature_dim=2,
    hidden=8,
    layers=1,
    dropout=0.0,
    nhead=2,
)
MODEL_CONTRACT = MODEL_CONTRACT_VALUE.to_document()
MODEL_CONFIG = ModelConfig.from_manifest(MODEL_CONTRACT_VALUE.model_config)


def test_worker_event_parser_rejects_duplicate_json_keys():
    encoded = encode_event(
        job_id="11111111-1111-4111-8111-111111111111",
        attempt=1,
        attempt_id="22222222-2222-4222-8222-222222222222",
        sequence=1,
        event_type="ready",
        payload={"pid": 1, "nextOrdinal": 0, "inputRevision": 0},
    )
    duplicate = encoded.replace(
        b'"contract":',
        b'"contract":"transformer-worker","contract":',
        1,
    )

    with pytest.raises(WorkerContractError, match="not valid UTF-8 JSON"):
        parse_event(duplicate)


def _data_contract(
    data_contract_sha256: str = DATA_CONTRACT_SHA256,
    *,
    identity: str = "test.learning-dataset",
) -> dict:
    return {
        "identity": identity,
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": data_contract_sha256,
        "seqLen": 2,
        "featureDim": 2,
    }


def _semantic_digests(
    data_contract_sha256: str = DATA_CONTRACT_SHA256,
) -> dict:
    return MODEL_CONTRACT_VALUE.digests(data_contract_sha256)


def _checkpoint_metadata(
    *,
    job_id: str,
    train_config: TrainConfig,
    data_contract_sha256: str = DATA_CONTRACT_SHA256,
    data_contract_identity: str = "test.learning-dataset",
    initialization: dict | None = None,
) -> dict:
    return {
        "format": "transformer-checkpoint-v6",
        "serviceVersion": "test",
        "generation": 1,
        "jobId": job_id,
        "dataContract": _data_contract(
            data_contract_sha256,
            identity=data_contract_identity,
        ),
        "modelContract": MODEL_CONTRACT,
        "semanticDigests": _semantic_digests(data_contract_sha256),
        "trainingConfig": train_config_to_manifest(train_config),
        "diagnostics": train_config.diagnostics.to_document(),
        "selection": {
            "enabled": False,
            "modelContractSha256": _semantic_digests(
                data_contract_sha256
            )["modelContractSha256"],
            "bestSelectionScore": None,
            "bestEpoch": None,
            "source": "last_epoch",
        },
        "initialization": initialization or {"kind": "random"},
        "jobConfigSha256": "a" * 64,
        "manifestSha256": MANIFEST_SHA256,
        "progress": {
            "completedEpochs": 1,
            "globalStep": 1,
            "trainingComplete": True,
        },
    }


def _write_input(
    path: Path,
    rows,
    *,
    fit: bool,
    range_ordinal: int = 0,
) -> None:
    schema = canonical_input_schema(
        "fit" if fit else "predict",
        SOURCE_ENCODING,
        seq_len=2,
        feature_dim=2,
        target_contract=MODEL_CONTRACT_VALUE.target_contract,
    )
    native_rows = [row[index:index + 2] for row in rows for index in (0, 2)]
    offsets = [[index * 2, index * 2 + 1] for index in range(len(rows))]
    arrays = [
        pa.array([range_ordinal], type=schema.field("rangeOrdinal").type),
        pa.array([0], type=schema.field("exampleOffset").type),
        pa.array([{
            "b0": {
                "nativeRows": native_rows,
                "observationOffsets": offsets,
            },
        }], type=schema.field("features").type),
    ]
    if fit:
        arrays.append(pa.array(
            [[
                [0.2]
                for _ in rows
            ]],
            type=schema.field("tgt").type,
        ))
    table = pa.Table.from_arrays(arrays, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)


def _input_manifest(
    path: Path,
    ordinal: int,
    rows: int,
    *,
    fit: bool,
    data_contract_sha256: str = DATA_CONTRACT_SHA256,
) -> dict:
    return {
        "schemaId": (
            "transformer.indexed-feature-blocks.fit.v1"
            if fit
            else "transformer.indexed-feature-blocks.predict.v1"
        ),
        "ordinal": ordinal,
        "commitRevision": ordinal + 1,
        "dataContractSha256": data_contract_sha256,
        "chunks": 1,
        "logicalRows": rows,
        "nativeRows": [rows * 2],
        "firstRangeOrdinal": ordinal,
        "firstExampleOffset": 0,
        "lastRangeOrdinal": ordinal,
        "nextExampleOffset": rows,
        "batches": 1,
        "artifact": _artifact(path),
    }


def test_worker_verifies_an_immutable_input_receipt_once(tmp_path, monkeypatch):
    input_path = tmp_path / "input.arrow"
    _write_input(input_path, [[1.0, 2.0, 3.0, 4.0]], fit=False)
    item = _input_manifest(input_path, 0, 1, fit=False)
    digest_calls = []
    sha256_file = worker_artifacts._sha256_file

    def counted_sha256(path):
        digest_calls.append(path)
        return sha256_file(path)

    monkeypatch.setattr(worker_artifacts, "_sha256_file", counted_sha256)
    committed = worker_artifacts.CommittedInputArtifacts()

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
            "--contract-version=12",
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
                    "detail": None,
                },
        )
    ]
    assert marker.encode() not in result.stdout


def test_closed_predict_worker_uses_data_digest_and_publishes_one_result(
    tmp_path,
):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    model_config = MODEL_CONFIG
    train_config = TrainConfig(
        batch_size=2,
        epochs=1,
    )
    source = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = build_model(
        model_config,
        source,
        None,
        torch.device("cpu"),
        MODEL_CONTRACT_VALUE,
    )
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(
        str(checkpoint),
        model,
        metadata=_checkpoint_metadata(
            job_id=job_id,
            train_config=train_config,
            data_contract_identity="test.opaque-envelope-renamed",
        ),
    )
    input_path = tmp_path / "input.arrow"
    _write_input(input_path, [[1.0, 2.0, 3.0, 4.0]], fit=False)

    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 12,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "predict",
        "predictionColumn": "predictions",
        "device": {"kind": "cpu"},
        "sourceEncoding": SOURCE_ENCODING,
        "inputs": [_input_manifest(input_path, 0, 1, fit=False)],
        "inputRevision": 1,
        "inputClosed": True,
        "manifestSha256": MANIFEST_SHA256,
        "workspace": {"root": str(workspace)},
        "model": {
            "checkpoint": _checkpoint_artifact(checkpoint),
        },
        "dataContract": _data_contract(),
        "modelContract": MODEL_CONTRACT,
        "semanticDigests": _semantic_digests(),
        "jobConfigSha256": "a" * 64,
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
    assert output_table.schema.field(0).type == pa.list_(pa.float32(), 1)


def test_closed_fit_worker_commits_global_epoch_checkpoint_and_result(tmp_path):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    train_config = TrainConfig(
        batch_size=2,
        epochs=1,
        seed=7,
        deterministic=True,
    )
    inputs = []
    for ordinal, rows in enumerate((
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]],
    )):
        input_path = tmp_path / f"input-{ordinal}.arrow"
        _write_input(input_path, rows, fit=True, range_ordinal=ordinal)
        inputs.append(_input_manifest(input_path, ordinal, len(rows), fit=True))

    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 12,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "fit",
        "device": {"kind": "cpu"},
        "sourceEncoding": SOURCE_ENCODING,
        "inputs": inputs,
        "inputRevision": 2,
        "inputClosed": True,
        "manifestSha256": MANIFEST_SHA256,
        "workspace": {"root": str(workspace)},
        "model": {
            "label": "returns.daily",
        },
        "initialization": {"kind": "random"},
        "training": train_config_to_manifest(train_config),
        "diagnostics": {
            "schemaVersion": 1,
            "gradientInteractions": None,
        },
        "dataContract": _data_contract(),
        "modelContract": MODEL_CONTRACT,
        "semanticDigests": _semantic_digests(),
        "jobConfigSha256": "a" * 64,
        "recovery": None,
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
    assert checkpoint_event["checkpointSerializationMs"] >= 0
    assert checkpoint_event["progress"] == {
        "completedEpochs": 1,
        "globalStep": checkpoint_event["metrics"]["step"],
        "trainingComplete": True,
    }
    assert validate_document(
        checkpoint_event["metrics"],
        "training-metrics",
    )["epoch"] == 1
    assert checkpoint_event["metrics"]["trainingBatchesCompleted"] == 2
    assert checkpoint_event["metrics"]["optimizerUpdatesApplied"] == 2
    assert checkpoint_event["metrics"]["optimizerUpdatesSkipped"] == 0
    assert checkpoint_event["metrics"]["finiteGradientBatches"] == 2
    assert checkpoint_event["metrics"]["nonFiniteGradientBatches"] == 0

    result_manifest = load_document(
        workspace / "worker-result.json",
        "result-manifest",
    )
    assert validate_document(result_manifest, "result-manifest") is result_manifest
    assert result_manifest["inputRevision"] == 2
    assert result_manifest["manifestSha256"] == MANIFEST_SHA256
    assert result_manifest["artifacts"] == []
    assert result_manifest["checkpointMetadata"]["dataContract"] == _data_contract()
    assert result_manifest["checkpointMetadata"]["initialization"] == {
        "kind": "random"
    }
    assert Path(result_manifest["checkpoint"]["path"]).is_file()
    assert result_manifest["checkpointSerializationMs"] >= 0


@pytest.mark.parametrize(
    (
        "parent_data_digest",
        "current_data_digest",
        "parent_data_identity",
        "expected_error",
    ),
    [
        (
            DATA_CONTRACT_SHA256,
            DATA_CONTRACT_SHA256,
            "test.learning-dataset",
            None,
        ),
        (
            DATA_CONTRACT_SHA256,
            DATA_CONTRACT_SHA256,
            "test.opaque-envelope-renamed",
            None,
        ),
        (
            "e" * 64,
            DATA_CONTRACT_SHA256,
            "test.learning-dataset",
            "parent checkpoint contract differs from the fit job",
        ),
    ],
)
def test_published_model_initialization_requires_exact_data_contract_digest(
    tmp_path,
    parent_data_digest,
    current_data_digest,
    parent_data_identity,
    expected_error,
):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    model_config = MODEL_CONFIG
    source = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    parent_model = build_model(
        model_config,
        source,
        None,
        torch.device("cpu"),
        MODEL_CONTRACT_VALUE,
    )
    with torch.no_grad():
        for parameter in parent_model.parameters():
            parameter.fill_(0.125)
    parent_path = tmp_path / "parent.pth"
    save_checkpoint(
        str(parent_path),
        parent_model,
        metadata=_checkpoint_metadata(
            job_id=job_id,
            train_config=TrainConfig(epochs=5),
            data_contract_sha256=parent_data_digest,
            data_contract_identity=parent_data_identity,
        ),
    )
    parent_artifact = _artifact(parent_path)
    input_path = tmp_path / "input.arrow"
    _write_input(
        input_path,
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        fit=True,
    )
    train_config = TrainConfig(
        lr=1e-20,
        batch_size=2,
        epochs=1,
        seed=7,
        deterministic=True,
    )
    initialization = published_model_initialization(
        "mdl_parent",
        parent_artifact["sha256"],
        _semantic_digests(parent_data_digest),
        _semantic_digests(current_data_digest),
    )
    manifest = {
        "contract": "transformer-worker",
        "protocolVersion": 12,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "fit",
        "device": {"kind": "cpu"},
        "sourceEncoding": SOURCE_ENCODING,
        "inputs": [
            _input_manifest(
                input_path,
                0,
                2,
                fit=True,
                data_contract_sha256=current_data_digest,
            )
        ],
        "inputRevision": 1,
        "inputClosed": True,
        "manifestSha256": MANIFEST_SHA256,
        "workspace": {"root": str(workspace)},
        "model": {
            "label": "returns.daily.fine-tuned",
            "parentCheckpoint": _checkpoint_artifact(parent_path),
        },
        "initialization": initialization,
        "training": train_config_to_manifest(train_config),
        "diagnostics": {
            "schemaVersion": 1,
            "gradientInteractions": None,
        },
        "dataContract": _data_contract(current_data_digest),
        "modelContract": MODEL_CONTRACT,
        "semanticDigests": _semantic_digests(current_data_digest),
        "jobConfigSha256": "a" * 64,
        "recovery": None,
    }
    manifest_path = workspace / "worker-command.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_worker(manifest_path, job_id, attempt_id, timeout=45)

    if expected_error is not None:
        assert result.returncode == 1
        events = [
            parse_event(line)
            for line in result.stdout.splitlines(keepends=True)
        ]
        assert events[-1]["type"] == "error"
        assert events[-1]["payload"] == {
            "code": "MODEL_SCHEMA_MISMATCH",
            "message": expected_error,
            "detail": None,
        }
        assert not (workspace / "worker-result.json").exists()
        return

    assert result.returncode == 0, result.stderr.decode()
    events = [
        parse_event(line) for line in result.stdout.splitlines(keepends=True)
    ]
    assert events[2]["type"] == "checkpoint"
    assert events[2]["payload"]["progress"]["globalStep"] == 1
    result_manifest = load_document(
        workspace / "worker-result.json",
        "result-manifest",
    )
    expected_lineage = initialization
    assert result_manifest["checkpointMetadata"]["initialization"] == (
        expected_lineage
    )
    parent_checkpoint = load_checkpoint(str(parent_path), "cpu")
    child_checkpoint = load_checkpoint(
        result_manifest["checkpoint"]["path"],
        "cpu",
    )
    assert child_checkpoint["metadata"]["initialization"] == expected_lineage
    assert child_checkpoint["metadata"]["dataContract"] == _data_contract(
        current_data_digest
    )
    for name, parent_value in parent_checkpoint["state_dict"].items():
        torch.testing.assert_close(
            child_checkpoint["state_dict"][name],
            parent_value,
            rtol=0,
            atol=0,
        )


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
        source_encoding=SOURCE_ENCODING,
        model_config=MODEL_CONFIG,
        training_config=None,
        data_contract=_data_contract(),
        model_contract=MODEL_CONTRACT,
        semantic_digests=_semantic_digests(),
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
        def mark_attempt_worker_ready(self, *args, **kwargs):
            return None

        def update_progress(self, job_id, progress, *, attempt_id):
            assert (job_id, progress, attempt_id) == (
                job.job_id,
                {"ordinal": 0, "rows": 1},
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
                payload={"progress": {"ordinal": 0, "rows": 1}},
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


@pytest.mark.parametrize("changed_field", ["jobConfigSha256", "semanticDigests"])
def test_service_rejects_result_manifest_with_changed_semantic_fence(
    tmp_path,
    changed_field,
):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    result_path = tmp_path / "worker-result.json"
    semantic_digests = _semantic_digests()
    job = SimpleNamespace(
        job_id=job_id,
        attempt=1,
        attempt_id=attempt_id,
        operation="predict",
        config_hash="a" * 64,
        semantic_digests=semantic_digests,
    )
    result = {
        "contract": "transformer-worker",
        "protocolVersion": 12,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "operation": "predict",
        "inputRevision": 1,
        "manifestSha256": MANIFEST_SHA256,
        "jobConfigSha256": "a" * 64,
        "semanticDigests": semantic_digests,
        "artifacts": [],
    }
    if changed_field == "jobConfigSha256":
        result[changed_field] = "e" * 64
    else:
        result[changed_field] = {
            **semantic_digests,
            "objectiveSha256": "e" * 64,
        }
    encoded = json.dumps(result).encode("utf-8")
    result_path.write_bytes(encoded)
    artifact = {
        "path": str(result_path),
        "byteCount": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    ledger = SimpleNamespace(get_job=lambda _job_id: {
        "input_state": InputState.CLOSED.value,
        "input_revision": 1,
        "manifest_sha256": MANIFEST_SHA256,
    })
    spool = SimpleNamespace(
        attempt_result_manifest_path=lambda _job_id, _attempt: str(result_path)
    )
    runner = WorkerSubprocessRunner(
        object(),
        ledger,
        spool,
        logger=object(),
        python_executable=sys.executable,
    )

    with pytest.raises(WorkerSubprocessError) as raised:
        runner._load_result_manifest(job, object(), artifact)  # type: ignore[arg-type]

    assert raised.value.code is ErrorCode.WORKER_PROTOCOL_VIOLATION


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
        source_encoding=SOURCE_ENCODING,
        model_config=MODEL_CONFIG,
        training_config=TrainConfig(),
        data_contract=_data_contract(),
        model_contract=MODEL_CONTRACT,
        semantic_digests=_semantic_digests(),
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

        def mark_attempt_worker_ready(self, *args, **kwargs):
            return None

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
        source_encoding=SOURCE_ENCODING,
        model_config=MODEL_CONFIG,
        training_config=TrainConfig(),
        data_contract=_data_contract(),
        model_contract=MODEL_CONTRACT,
        semantic_digests=_semantic_digests(),
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
        schema_id="transformer.indexed-feature-blocks.fit.v1",
        data_contract_sha256=DATA_CONTRACT_SHA256,
        chunks=1,
        rows=2,
        native_rows=(4,),
        first_range_ordinal=0,
        first_example_offset=0,
        last_range_ordinal=0,
        next_example_offset=2,
        batches=1,
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
                "total_chunks": 1,
                "total_rows": 2,
                "total_native_rows": [4],
                "range_count": 1,
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
    assert errors.empty(), errors.queue[0].message
    assert [message["type"] for message in messages] == [
        "input",
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
            "--contract-version=12",
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


def _checkpoint_artifact(path: Path) -> dict:
    artifact = _artifact(path)
    return {
        "path": artifact["path"],
        "format": "transformer-checkpoint-v6",
        "byteCount": artifact["byteCount"],
        "checkpointSha256": artifact["sha256"],
    }
