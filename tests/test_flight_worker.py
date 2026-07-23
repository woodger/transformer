from dataclasses import replace
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pytest

from app.database.models import JobAttempt
from app.flight.arrow import schema_fingerprint
from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CANCEL_ACTION,
    CONTRACT_NAME,
    ErrorCode,
    FIT_SCHEMA_ID,
    JobState,
    PREDICT_SCHEMA_ID,
    STATUS_ACTION,
)
from app.flight.contract import validate_action_request
from app.flight.coordinator import JobCoordinator
from app.flight.ledger import Ledger
from app.flight.observability import OperationalMetrics
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool
from app.flight.worker import WorkerPool
from app.training.run_config import ModelConfig, TrainConfig


PREDICT_HELPER = r"""
import sys
import pyarrow as pa
import pyarrow.ipc as ipc

mode = sys.argv[1]
index = 0
while True:
    header = sys.stdin.buffer.read(8)
    if not header:
        break
    size = int.from_bytes(header, "big")
    if size == 0:
        break
    payload = sys.stdin.buffer.read(size)
    table = ipc.RecordBatchFileReader(pa.BufferReader(payload)).read_all()
    if mode == "bad-second" and index == 1:
        output = b"not-arrow"
    else:
        values = [[float(i) for i in range(6)] for _ in range(table.num_rows)]
        result = pa.table({"out": pa.array(values, type=pa.list_(pa.float32()))})
        sink = pa.BufferOutputStream()
        with ipc.new_file(sink, result.schema) as writer:
            writer.write_table(result)
        output = sink.getvalue().to_pybytes()
    sys.stdout.buffer.write(len(output).to_bytes(8, "big"))
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    index += 1
print(f"frames={index}", file=sys.stderr, flush=True)
"""


FIT_HELPER = r"""
import json
import os
import sys
import torch

checkpoint_path, metrics_path = sys.argv[1:3]
model_config = json.loads(sys.argv[3])
train_config = json.loads(sys.argv[4])
data_schema = json.loads(sys.argv[5])
expected_frames = int(sys.argv[6])
frames = 0
while True:
    header = sys.stdin.buffer.read(8)
    if not header:
        break
    size = int.from_bytes(header, "big")
    if size == 0:
        break
    remaining = size
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            raise RuntimeError("truncated frame")
        remaining -= len(chunk)
    frames += 1
if frames != expected_frames:
    print(f"expected {expected_frames} frames, got {frames}", file=sys.stderr)
    raise SystemExit(9)
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w", encoding="utf-8") as metrics:
    for frame in range(1, frames + 1):
        metrics.write(json.dumps({"frame": frame, "epoch": 1, "loss": frame / 10}) + "\n")
torch.save({
    "format": "transformer-checkpoint-v2",
    "version": "worker-test",
    "state_dict": {},
    "model_config": model_config,
    "train_config": train_config,
    "data_schema": data_schema,
    "extra": {"checkpoint_selection": {
        "monitor": train_config["monitor"],
        "monitor_min_improvement": train_config["monitor_min_improvement"],
        "best_monitor": 0.1,
        "best_frame": frames,
        "best_epoch": 1,
        "baseline_passed": True,
        "source": "best_monitor",
    }},
}, checkpoint_path)
print(f"frames={frames}", flush=True)
"""


_POSTGRES_LEDGER = None


@pytest.fixture(autouse=True)
def _use_postgres_ledger(postgres_ledger):
    global _POSTGRES_LEDGER
    _POSTGRES_LEDGER = postgres_ledger
    try:
        yield
    finally:
        _POSTGRES_LEDGER = None


def service_config(tmp_path, **overrides):
    base = FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        allow_plaintext=True,
        disk_min_free_bytes=1,
        cpu_capacity=2,
        cancel_grace_seconds=0.1,
        shutdown_drain_seconds=0.2,
        subprocess_timeout_seconds=5.0,
    )
    return replace(base, **overrides).validate()


def components(tmp_path, *, config=None, **pool_options):
    config = config or service_config(tmp_path)
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    ledger = _POSTGRES_LEDGER
    pool = WorkerPool(config, ledger, spool, **pool_options)
    return config, spool, ledger, pool


def model_config(feature_dim=None):
    return ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        context_mode="relaxed",
        feature_dim=feature_dim,
    )


def train_config(**overrides):
    values = {
        "lr": 0.001,
        "batch_size": 2,
        "epochs": 1,
        "patience": 0,
        "loss_stage": 1,
        "loss_schedule": "none",
        "stage_size": 1,
        "use_amp": False,
        "weight_decay": 0.0025,
        "monitor": "loss",
        "monitor_min_improvement": 0.0,
        "save_best_checkpoint": False,
        "seed": 7,
        "deterministic": True,
    }
    values.update(overrides)
    return TrainConfig(**values)


def predict_schema():
    return pa.schema([("src", pa.list_(pa.float32()))])


def fit_schema():
    return pa.schema([
        ("src", pa.list_(pa.float32())),
        ("tgt", pa.list_(pa.float32())),
    ])


def predict_batch(rows):
    return pa.record_batch(
        {"src": pa.array(rows, type=pa.list_(pa.float32()))},
        schema=predict_schema(),
    )


def fit_batch(rows):
    return pa.record_batch(
        {
            "src": pa.array(rows, type=pa.list_(pa.float32())),
            "tgt": pa.array(
                [[0.0, 0.0, 0.0, 0.0, 0.2, 1.0] for _ in rows],
                type=pa.list_(pa.float32()),
            ),
        },
        schema=fit_schema(),
    )


def commit_input(ledger, spool, job, ordinal, batches):
    schema = fit_schema() if job["operation"] == "fit" else predict_schema()
    path = spool.input_path(job["job_id"], ordinal)
    spool.ensure_parent(path)
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
    data = Path(path).read_bytes()
    payload_id = str(uuid.uuid4())
    token = f"upload-{job['job_id']}-{ordinal}"
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=ordinal,
        upload_token=token,
        temporary_path=spool.relative_path(path) + ".reserved",
    )
    rows = sum(batch.num_rows for batch in batches)
    ledger.commit_input(
        upload_token=token,
        relative_path=spool.relative_path(path),
        schema_id=FIT_SCHEMA_ID if job["operation"] == "fit" else PREDICT_SCHEMA_ID,
        rows=rows,
        batches=len(batches),
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        schema_fingerprint=schema_fingerprint(schema),
        source_width=4 if rows else None,
        feature_dim=2 if rows else None,
        max_payloads=100,
        max_job_bytes=1024 * 1024 * 1024,
    )
    return {
        "payloadId": payload_id,
        "ordinal": ordinal,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def seal_and_queue(ledger, job, manifest, *, device="cpu"):
    ledger.seal_job(
        job["job_id"],
        manifest_hash=hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        manifest=manifest,
        source_width=4 if manifest else None,
        feature_dim=2 if manifest else None,
    )
    ledger.queue_job(job["job_id"], selected_device=device)


def create_fit_job(ledger, *, requested_device="cpu"):
    return ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device=requested_device,
        prediction_column="out",
        config_hash="a" * 64,
        model_label="returns.daily",
        model_config=model_config(),
        training_config=train_config(),
    )


def publish_seed_model(ledger, spool):
    producer = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="b" * 64,
        model_label="seed",
        model_config=model_config(feature_dim=2),
        training_config=train_config(),
    )
    seal_and_queue(ledger, producer, [])
    running = ledger.claim_next_job("cpu", worker_id="seed-publisher")
    model_ref = f"mdl_seed_{uuid.uuid4().hex}"
    checkpoint = spool.model_checkpoint_path(model_ref)
    metadata_path = spool.model_metadata_path(model_ref)
    spool.atomic_write_bytes(checkpoint, b"server-owned-checkpoint")
    spool.atomic_write_json(metadata_path, {"model_config": model_config(2).to_dict()})
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    metadata = {
        "model_config": model_config(2).to_dict(),
        "train_config": train_config().to_dict(),
        "checkpoint": {},
    }
    ledger.publish_model(
        producer["job_id"],
        running["attempt"],
        model_ref=model_ref,
        label="seed",
        generation=None,
        checkpoint_path=spool.model_relative_path(checkpoint),
        metadata_path=spool.model_relative_path(metadata_path),
        sha256=digest,
        metadata=metadata,
        result={"modelRef": model_ref, "checkpoint": {}},
    )
    return model_ref


def create_predict_job(ledger, spool):
    model_ref = publish_seed_model(ledger, spool)
    return ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="c" * 64,
        input_model_ref=model_ref,
        model_config=model_config(feature_dim=2),
    )


def test_fit_argv_contains_exact_immutable_config_and_server_paths(tmp_path):
    config, spool, ledger, pool = components(tmp_path)
    job = create_fit_job(ledger)
    seal_and_queue(ledger, job, [])
    queued = ledger.get_job(job["job_id"])
    argv = pool.build_argv({**queued, "attempt": 1}, 1)

    assert argv == [
        sys.executable,
        pool._cli_path,
        "fit-stream",
        "--device",
        "cpu",
        "--checkpoint-out",
        spool.attempt_checkpoint_path(job["job_id"], 1),
        "--metrics-out",
        spool.attempt_metrics_path(job["job_id"], 1),
        "--max-frame-bytes",
        str(config.max_payload_bytes),
        "--input-spool-dir",
        spool.input_directory(job["job_id"]),
        "--input-frame-count",
        "0",
        "--seq-len",
        "2",
        "--hidden",
        "8",
        "--layers",
        "1",
        "--dropout",
        "0.0",
        "--nhead",
        "2",
        "--mode",
        "relaxed",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0025",
        "--batch-size",
        "2",
        "--epochs",
        "1",
        "--loss-stage",
        "1",
        "--loss-schedule",
        "none",
        "--stage-size",
        "1",
        "--patience",
        "0",
        "--monitor",
        "loss",
        "--monitor-min-improvement",
        "0.0",
        "--seed",
        "7",
        "--no-save-best-checkpoint",
        "--deterministic",
    ]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("ordinal", "sealed input ordinals are inconsistent"),
        ("path", "committed input path is invalid"),
        ("digest", "committed input digest is invalid"),
    ],
)
def test_worker_plan_rejects_untrusted_committed_input(
    tmp_path,
    monkeypatch,
    corruption,
    message,
):
    _, spool, ledger, pool = components(tmp_path)
    job = create_fit_job(ledger)
    manifest = [
        commit_input(
            ledger,
            spool,
            job,
            0,
            [fit_batch([[1.0, 2.0, 3.0, 4.0]])],
        )
    ]
    seal_and_queue(ledger, job, manifest)
    record = dict(ledger.list_inputs(job["job_id"])[0])
    if corruption == "ordinal":
        record["ordinal"] = 1
    elif corruption == "path":
        record["relative_path"] = spool.relative_path(
            spool.input_path(str(uuid.uuid4()), 0)
        )
    else:
        record["sha256"] = "0" * 64
    monkeypatch.setattr(ledger, "list_inputs", lambda _: [record])

    assert pool.run_once("cpu") is True

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == ErrorCode.INTERNAL.value
    assert failed["error_message"] == message


def test_worker_plan_rejects_noncanonical_selected_device(tmp_path):
    _, _, ledger, pool = components(tmp_path)
    job = create_fit_job(ledger)

    with pytest.raises(Exception, match="queued job has no selected device") as error:
        pool.build_argv(
            {**job, "attempt": 1, "selected_device": "auto"},
            1,
        )

    assert error.value.code == ErrorCode.INTERNAL


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("path", "resolved model checkpoint is unavailable"),
        ("digest", "resolved model checkpoint digest is invalid"),
    ],
)
def test_worker_plan_rejects_untrusted_model_generation(
    tmp_path,
    monkeypatch,
    corruption,
    message,
):
    _, spool, ledger, pool = components(tmp_path)
    job = create_predict_job(ledger, spool)
    model = ledger.get_model(
        job["input_model_ref"],
        owner_subject=job["owner_subject"],
    )
    if corruption == "path":
        model["checkpoint_path"] = (
            f"{model['model_ref']}/unexpected-checkpoint.pth"
        )
    else:
        model["sha256"] = "0" * 64
    monkeypatch.setattr(ledger, "get_model", lambda *_, **__: model)

    with pytest.raises(Exception, match=message) as error:
        pool.build_argv(
            {**job, "attempt": 1, "selected_device": "cpu"},
            1,
        )

    assert error.value.code == ErrorCode.INTERNAL


def test_predict_one_process_preserves_payload_boundaries_and_ordinals(tmp_path):
    calls = []
    defaults = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.Popen(argv, **kwargs)

    def hook(job, argv):
        defaults.append(argv)
        return [sys.executable, "-c", PREDICT_HELPER, "ok"]

    config, spool, ledger, pool = components(
        tmp_path,
        popen_factory=popen,
        argv_hook=hook,
    )
    job = create_predict_job(ledger, spool)
    manifest = [
        commit_input(
            ledger,
            spool,
            job,
            0,
            [
                predict_batch([[1.0, 2.0, 3.0, 4.0]]),
                predict_batch([[5.0, 6.0, 7.0, 8.0]]),
            ],
        ),
        commit_input(ledger, spool, job, 1, []),
    ]
    seal_and_queue(ledger, job, manifest)

    assert pool.run_once("cpu") is True

    finished = ledger.get_job(job["job_id"])
    outputs = ledger.list_outputs(job["job_id"])
    assert finished["state"] == JobState.SUCCEEDED.value
    assert [(item["ordinal"], item["rows"]) for item in outputs] == [(0, 2), (1, 0)]
    assert len(calls) == 1
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["start_new_session"] is True
    launch_argv = calls[0][0]
    assert launch_argv[0] == sys.executable
    assert launch_argv[1] == os.path.join(
        Path(pool._cli_path).parents[1],
        "app",
        "flight",
        "process_supervisor.py",
    )
    assert launch_argv[2:4] == [str(os.getpid()), "--"]
    assert launch_argv[4:] == [sys.executable, "-c", PREDICT_HELPER, "ok"]
    model = ledger.get_model(
        job["input_model_ref"],
        owner_subject=job["owner_subject"],
    )
    assert defaults == [(
        sys.executable,
        pool._cli_path,
        "predict-stream",
        "--device",
        "cpu",
        "--checkpoint",
        spool.model_absolute_path(model["checkpoint_path"]),
        "--pred-col",
        "out",
        "--max-frame-bytes",
        str(config.max_payload_bytes),
    )]


def test_malformed_second_prediction_publishes_no_partial_outputs(tmp_path):
    _, spool, ledger, pool = components(
        tmp_path,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            PREDICT_HELPER,
            "bad-second",
        ],
    )
    job = create_predict_job(ledger, spool)
    manifest = [
        commit_input(ledger, spool, job, 0, [predict_batch([[1.0, 2.0, 3.0, 4.0]])]),
        commit_input(ledger, spool, job, 1, [predict_batch([[5.0, 6.0, 7.0, 8.0]])]),
    ]
    seal_and_queue(ledger, job, manifest)

    pool.run_once("cpu")

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "MALFORMED_OUTPUT"
    assert ledger.list_outputs(job["job_id"]) == []
    assert not os.path.exists(spool.attempt_output_path(job["job_id"], 1, 0))


def test_prediction_output_disk_exhaustion_has_stable_error_code(
    tmp_path,
    monkeypatch,
):
    _, spool, ledger, pool = components(
        tmp_path,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            PREDICT_HELPER,
            "ok",
        ],
    )
    job = create_predict_job(ledger, spool)
    manifest = [
        commit_input(
            ledger,
            spool,
            job,
            0,
            [predict_batch([[1.0, 2.0, 3.0, 4.0]])],
        )
    ]
    seal_and_queue(ledger, job, manifest)
    monkeypatch.setattr(
        spool,
        "create_temporary",
        lambda destination: (_ for _ in ()).throw(
            OSError(errno.EDQUOT, "injected quota exhaustion")
        ),
    )

    pool.run_once("cpu")

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "DISK_FULL"
    assert failed["error_message"] == "prediction output could not be persisted"
    assert ledger.list_outputs(job["job_id"]) == []


def test_log_disk_quota_exhaustion_is_classified_as_disk_full(
    tmp_path,
    monkeypatch,
):
    _, spool, _, pool = components(tmp_path)
    path = spool.attempt_stderr_path(str(uuid.uuid4()), 1)
    real_open = open

    def quota_exhausted(candidate, *args, **kwargs):
        if os.fspath(candidate) == path:
            raise OSError(errno.EDQUOT, "injected quota exhaustion")
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr("builtins.open", quota_exhausted)
    errors = queue.Queue()
    pool._drain_log(io.BytesIO(b"diagnostic"), path, errors)

    failure = errors.get_nowait()
    assert failure.code.value == "DISK_FULL"
    assert failure.message == "worker log could not be persisted"


def test_cleanup_failure_cannot_leave_malformed_prediction_running(
    tmp_path,
    monkeypatch,
):
    _, spool, ledger, pool = components(
        tmp_path,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            PREDICT_HELPER,
            "bad-second",
        ],
    )
    job = create_predict_job(ledger, spool)
    manifest = [
        commit_input(ledger, spool, job, 0, [predict_batch([[1.0, 2.0, 3.0, 4.0]])]),
        commit_input(ledger, spool, job, 1, [predict_batch([[5.0, 6.0, 7.0, 8.0]])]),
    ]
    seal_and_queue(ledger, job, manifest)
    original_remove = spool.remove
    calls = []

    def fail_attempt_cleanup(path):
        if path == spool.attempt_output_path(job["job_id"], 1, 0):
            calls.append(path)
            raise OSError(errno.EIO, "injected cleanup failure")
        return original_remove(path)

    monkeypatch.setattr(spool, "remove", fail_attempt_cleanup)

    pool.run_once("cpu")

    failed = ledger.get_job(job["job_id"])
    assert calls
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "MALFORMED_OUTPUT"
    assert ledger.list_outputs(job["job_id"]) == []


def test_cancel_registered_during_failure_cleanup_wins_terminal_race(
    tmp_path,
    monkeypatch,
):
    _, spool, ledger, pool = components(
        tmp_path,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            PREDICT_HELPER,
            "bad-second",
        ],
    )
    job = create_predict_job(ledger, spool)
    manifest = [
        commit_input(ledger, spool, job, 0, [predict_batch([[1.0, 2.0, 3.0, 4.0]])]),
        commit_input(ledger, spool, job, 1, [predict_batch([[5.0, 6.0, 7.0, 8.0]])]),
    ]
    seal_and_queue(ledger, job, manifest)
    original_remove = spool.remove
    cancelled = False

    def cancel_then_remove(path):
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            ledger.transition_job(job["job_id"], JobState.CANCELLING)
        return original_remove(path)

    monkeypatch.setattr(spool, "remove", cancel_then_remove)

    pool.run_once("cpu")

    finished = ledger.get_job(job["job_id"])
    assert cancelled is True
    assert finished["state"] == JobState.CANCELLED.value
    assert ledger.list_outputs(job["job_id"]) == []


def test_fit_two_inputs_publish_one_immutable_model_and_progress(tmp_path):
    def hook(job, argv):
        checkpoint = argv[argv.index("--checkpoint-out") + 1]
        metrics = argv[argv.index("--metrics-out") + 1]
        actual_model = model_config(feature_dim=2).to_dict()
        actual_train = train_config().to_dict()
        data_schema = {
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
        }
        return [
            sys.executable,
            "-c",
            FIT_HELPER,
            checkpoint,
            metrics,
            json.dumps(actual_model),
            json.dumps(actual_train),
            json.dumps(data_schema),
            "2",
        ]

    config, spool, ledger, pool = components(tmp_path, argv_hook=hook)
    job = create_fit_job(ledger)
    manifest = [
        commit_input(
            ledger,
            spool,
            job,
            0,
            [
                fit_batch([[1.0, 2.0, 3.0, 4.0]]),
                fit_batch([[5.0, 6.0, 7.0, 8.0]]),
            ],
        ),
        commit_input(ledger, spool, job, 1, [fit_batch([[9.0, 10.0, 11.0, 12.0]])]),
    ]
    seal_and_queue(ledger, job, manifest)

    pool.run_once("cpu")

    finished = ledger.get_job(job["job_id"])
    assert finished["state"] == JobState.SUCCEEDED.value
    assert finished["result"]["modelRef"].startswith("mdl_")
    assert "/" not in finished["result"]["modelRef"]
    checkpoint = finished["result"]["checkpoint"]
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
    assert checkpoint["trainConfig"]["weightDecay"] == 0.0025
    assert checkpoint["dataSchema"]["source"]["width"] == 4
    assert checkpoint["checkpointSelection"]["bestFrame"] == 2
    model = ledger.get_model(finished["result"]["modelRef"])
    assert model["metadata"]["model_config"]["feature_dim"] == 2
    assert os.path.isfile(spool.model_absolute_path(model["checkpoint_path"]))
    assert finished["progress"]["ordinal"] == 1
    status = JobCoordinator(config, ledger, spool).status(
        "inventory",
        job["job_id"],
        str(uuid.uuid4()),
    )
    assert set(status) == {
        "contract",
        "version",
        "requestId",
        "jobId",
        "operation",
        "state",
        "revision",
        "timestamps",
        "device",
        "committedInputs",
        "progress",
        "attempt",
        "error",
        "results",
        "pollAfterMs",
    }
    assert status["contract"] == "transformer-flight"
    assert status["version"] == 1
    assert status["state"] == JobState.SUCCEEDED.value
    assert status["results"] == {
        "outputs": [],
        "modelRef": finished["result"]["modelRef"],
        "checkpoint": checkpoint,
    }


def test_successful_fit_contract_requires_at_least_one_metrics_record(tmp_path):
    _, _, ledger, pool = components(tmp_path)
    job = create_fit_job(ledger)
    seal_and_queue(ledger, job, [])
    running = ledger.claim_next_job("cpu")
    finished = threading.Event()
    finished.set()
    errors = queue.Queue()

    pool._tail_metrics(running, [], finished, errors)

    failure = errors.get_nowait()
    assert failure.code.value == "MALFORMED_OUTPUT"
    assert "did not produce metrics" in failure.message


def test_metrics_tailer_performs_final_read_when_process_exits_during_stat(
    tmp_path,
    monkeypatch,
):
    _, spool, ledger, pool = components(tmp_path)
    job = create_fit_job(ledger)
    seal_and_queue(ledger, job, [])
    running = ledger.claim_next_job("cpu")
    metrics_path = Path(spool.attempt_metrics_path(job["job_id"], 1))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_bytes(b'{"frame":1')
    finished = threading.Event()
    errors = queue.Queue()
    real_stat = os.stat
    calls = 0

    def exit_during_first_stat(path, *args, **kwargs):
        nonlocal calls
        if os.fspath(path) == os.fspath(metrics_path):
            calls += 1
            if calls == 1:
                result = real_stat(path, *args, **kwargs)
                finished.set()
                return result
            if calls == 2:
                with metrics_path.open("ab") as stream:
                    stream.write(b',"epoch":1,"loss":0.1}\n')
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", exit_during_first_stat)

    pool._tail_metrics(running, [], finished, errors)

    assert calls >= 2
    assert errors.empty()
    assert ledger.get_job(job["job_id"])["progress"] == {
        "epoch": 1,
        "frame": 1,
        "loss": 0.1,
    }


def test_corrupt_fit_checkpoint_has_stable_subprocess_error(tmp_path):
    script = r"""
import json
import pathlib
import sys

checkpoint_path = pathlib.Path(sys.argv[1])
metrics_path = pathlib.Path(sys.argv[2])
while True:
    header = sys.stdin.buffer.read(8)
    if not header:
        break
    size = int.from_bytes(header, "big")
    if size == 0:
        break
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise RuntimeError("truncated frame")
metrics_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.write_text(json.dumps({"frame": 1, "epoch": 1, "loss": 0.1}) + "\n")
checkpoint_path.write_bytes(b"")
"""

    def hook(job, argv):
        return [
            sys.executable,
            "-c",
            script,
            argv[argv.index("--checkpoint-out") + 1],
            argv[argv.index("--metrics-out") + 1],
        ]

    _, spool, ledger, pool = components(tmp_path, argv_hook=hook)
    job = create_fit_job(ledger)
    manifest = [
        commit_input(
            ledger,
            spool,
            job,
            0,
            [fit_batch([[1.0, 2.0, 3.0, 4.0]])],
        )
    ]
    seal_and_queue(ledger, job, manifest)

    pool.run_once("cpu")

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "SUBPROCESS_FAILED"
    assert failed["error_message"] == "fit subprocess created an invalid checkpoint"


def test_nonzero_cuda_oom_has_stable_error_code(tmp_path):
    metrics = OperationalMetrics()
    _, spool, ledger, pool = components(
        tmp_path,
        metrics=metrics,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            "import sys; print('CUDA out of memory', file=sys.stderr); raise SystemExit(2)",
        ],
    )
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [], device="cuda")

    pool.run_once("cuda")

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "CUDA_OUT_OF_MEMORY"
    assert metrics.snapshot()["counters"]["cudaOutOfMemory"] == 1


def test_cuda_disconnect_has_stable_device_unavailable_error(tmp_path):
    _, spool, ledger, pool = components(
        tmp_path,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('CUDA driver shutting down', file=sys.stderr); "
                "raise SystemExit(3)"
            ),
        ],
    )
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [], device="cuda")

    pool.run_once("cuda")

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "DEVICE_UNAVAILABLE"
    assert failed["error_message"] == (
        "CUDA device became unavailable during execution"
    )


def test_timeout_terminates_process_group_and_marks_job_failed(tmp_path):
    config = service_config(
        tmp_path,
        subprocess_timeout_seconds=0.1,
        cancel_grace_seconds=0.05,
    )
    _, spool, ledger, pool = components(
        tmp_path,
        config=config,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [])

    pool.run_once("cpu")

    failed = ledger.get_job(job["job_id"])
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "SUBPROCESS_HUNG"


def test_post_spawn_setup_failure_kills_and_reaps_subprocess(
    tmp_path,
    monkeypatch,
):
    processes = []

    def popen(argv, **kwargs):
        process = subprocess.Popen(argv, **kwargs)
        processes.append(process)
        return process

    _, spool, ledger, pool = components(
        tmp_path,
        popen_factory=popen,
        argv_hook=lambda job, argv: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [])
    monkeypatch.setattr(
        ledger,
        "set_attempt_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.ENOSPC, "injected state-database setup failure")
        ),
    )

    pool.run_once("cpu")

    assert ledger.get_job(job["job_id"])["state"] == JobState.FAILED.value
    assert processes and processes[0].poll() is not None


def test_inherited_pipes_after_leader_exit_remain_under_worker_timeout(tmp_path):
    config = service_config(
        tmp_path,
        # Leave enough time for the parent-death supervisor and test helper to
        # start; the inherited descendant must still be bounded by the same
        # worker deadline once the CLI leader exits.
        subprocess_timeout_seconds=0.5,
        cancel_grace_seconds=0.05,
    )
    script = (
        "import subprocess,sys; "
        "sys.stdin.buffer.read(8); "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(f'grandchild={child.pid}', file=sys.stderr, flush=True)"
    )
    _, spool, ledger, pool = components(
        tmp_path,
        config=config,
        argv_hook=lambda job, argv: [sys.executable, "-c", script],
    )
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [])

    started = time.monotonic()
    pool.run_once("cpu")
    elapsed = time.monotonic() - started

    failed = ledger.get_job(job["job_id"])
    assert elapsed < 2.0
    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == "SUBPROCESS_HUNG"
    stderr = Path(spool.attempt_stderr_path(job["job_id"], 1)).read_text()
    grandchild = int(stderr.split("grandchild=", 1)[1].splitlines()[0])
    state_path = Path(f"/proc/{grandchild}/stat")
    if state_path.exists():
        assert state_path.read_text().split()[2] == "Z"


def test_cancel_and_status_loopback_terminate_running_process_group(tmp_path):
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(f'grandchild={child.pid}', file=sys.stderr, flush=True); "
        "time.sleep(30)"
    )
    config, spool, ledger, pool = components(
        tmp_path,
        argv_hook=lambda job, argv: [sys.executable, "-c", script],
    )
    coordinator = JobCoordinator(
        config,
        ledger,
        spool,
        cancel_notifier=pool.notify_cancel,
    )
    server = TransformerFlightServer(
        config,
        coordinator,
        {"secret": "inventory"},
    )
    client = flight.FlightClient(("localhost", server.port))

    def action(name, **fields):
        document = {
            "contract": CONTRACT_NAME,
            "version": 1,
            "requestId": str(uuid.uuid4()),
            **fields,
        }
        options = flight.FlightCallOptions(
            headers=[(b"authorization", b"Bearer secret")],
            timeout=5.0,
        )
        results = list(client.do_action(
            flight.Action(name, json.dumps(document).encode()),
            options=options,
        ))
        assert len(results) == 1
        return json.loads(results[0].body.to_pybytes())

    job = create_fit_job(ledger, requested_device="cuda")
    seal_and_queue(ledger, job, [], device="cuda")
    runner = threading.Thread(target=pool.run_once, args=("cuda",))
    runner.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            status = action(STATUS_ACTION, jobId=job["job_id"])
            stderr_path = spool.attempt_stderr_path(job["job_id"], 1)
            if (
                status["state"] == JobState.RUNNING.value
                and os.path.exists(stderr_path)
            ):
                text = Path(stderr_path).read_text(errors="replace")
                if "grandchild=" in text:
                    break
            time.sleep(0.02)
        else:
            raise AssertionError("worker subprocess did not start")

        grandchild = int(text.split("grandchild=", 1)[1].splitlines()[0])
        response = action(
            CANCEL_ACTION,
            idempotencyKey="cancel-running-worker",
            jobId=job["job_id"],
        )
        runner.join(5)

        assert response["state"] == JobState.CANCELLING.value
        assert not runner.is_alive()
        status = action(STATUS_ACTION, jobId=job["job_id"])
        assert status["state"] == JobState.CANCELLED.value
        assert status["error"] is None
        assert status["pollAfterMs"] == 0
        finished = ledger.get_job(job["job_id"])
        assert finished["error_code"] is None
        assert finished["error_message"] is None
        with ledger.connection() as connection:
            attempt = connection.get(JobAttempt, (job["job_id"], 1))
        assert attempt.status == JobState.CANCELLED.value
        assert attempt.error_code is None
        assert attempt.error_message is None
        state_path = Path(f"/proc/{grandchild}/stat")
        if state_path.exists():
            # A short-lived zombie is already terminated and cannot consume CPU/GPU.
            assert state_path.read_text().split()[2] == "Z"
    finally:
        if runner.is_alive():
            pool.notify_cancel(job["job_id"])
            runner.join(5)
        assert not runner.is_alive()
        client.close()
        server.shutdown()


def test_cancel_escalates_to_sigkill_for_term_resistant_process_group(tmp_path):
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    script = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}], "
        "stdout=subprocess.PIPE); "
        "child.stdout.readline(); "
        "print(f'grandchild={child.pid}', file=sys.stderr, flush=True); "
        "time.sleep(30)"
    )
    sent_signals = []

    def signal_group(pgid, signum):
        sent_signals.append(signum)
        os.killpg(pgid, signum)

    config = service_config(tmp_path, cancel_grace_seconds=0.05)
    _, spool, ledger, pool = components(
        tmp_path,
        config=config,
        argv_hook=lambda job, argv: [sys.executable, "-c", script],
        signal_group=signal_group,
    )
    coordinator = JobCoordinator(
        config,
        ledger,
        spool,
        cancel_notifier=pool.notify_cancel,
    )
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [])
    runner = threading.Thread(target=pool.run_once, args=("cpu",))
    runner.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            stderr_path = spool.attempt_stderr_path(job["job_id"], 1)
            if os.path.exists(stderr_path):
                text = Path(stderr_path).read_text(errors="replace")
                if "grandchild=" in text:
                    break
            time.sleep(0.02)
        else:
            raise AssertionError("term-resistant subprocess group did not start")

        grandchild = int(text.split("grandchild=", 1)[1].splitlines()[0])
        cancel = {
            "contract": CONTRACT_NAME,
            "version": 1,
            "requestId": str(uuid.uuid4()),
            "idempotencyKey": "cancel-term-resistant-worker",
            "jobId": job["job_id"],
        }
        coordinator.cancel(
            "inventory",
            validate_action_request(CANCEL_ACTION, cancel),
            cancel,
        )
        runner.join(5)

        assert not runner.is_alive()
        assert sent_signals[:2] == [signal.SIGTERM, signal.SIGKILL]
        assert ledger.get_job(job["job_id"])["state"] == JobState.CANCELLED.value
        state_path = Path(f"/proc/{grandchild}/stat")
        if state_path.exists():
            assert state_path.read_text().split()[2] == "Z"
    finally:
        if runner.is_alive():
            pool.notify_cancel(job["job_id"])
            runner.join(5)
        assert not runner.is_alive()


def test_pool_exposes_configured_cpu_lanes_and_exactly_one_cuda_lane(tmp_path):
    config, _, _, pool = components(tmp_path)
    assert pool.lane_counts == {"cpu": config.cpu_capacity, "cuda": 1}


def test_stop_claiming_preserves_queued_job_for_restart(tmp_path):
    _, spool, ledger, pool = components(tmp_path)
    job = create_predict_job(ledger, spool)
    seal_and_queue(ledger, job, [], device="cpu")

    pool.stop_claiming()

    assert pool.run_once("cpu", worker_id="shutdown-race") is False
    assert ledger.get_job(job["job_id"])["state"] == JobState.QUEUED.value


def test_unexpected_attempt_error_does_not_permanently_kill_lane(
    tmp_path,
    monkeypatch,
):
    _, _, _, pool = components(tmp_path)
    calls = []

    def flaky_claim(device, job_id, worker_id):
        calls.append((device, job_id, worker_id))
        if len(calls) == 1:
            raise RuntimeError("injected lane failure")
        pool.stop_claiming()
        return False

    monkeypatch.setattr(pool, "_claim_and_execute", flaky_claim)
    pool._queues["cpu"].put("first-job")
    pool._queues["cpu"].put("second-job")
    lane = threading.Thread(target=pool._lane, args=("cpu", "test-lane"))

    lane.start()
    lane.join(2)

    assert not lane.is_alive()
    assert len(calls) == 2


def test_cuda_lane_runs_queued_jobs_fifo_without_overlap(tmp_path):
    execution_log = tmp_path / "cuda-execution.log"
    script = (
        "import pathlib,sys,time; "
        "path=pathlib.Path(sys.argv[1]); job=sys.argv[2]; "
        "path.open('a').write('start '+job+'\\n'); "
        "time.sleep(0.15); "
        "path.open('a').write('end '+job+'\\n')"
    )

    def hook(job, argv):
        return [
            sys.executable,
            "-c",
            script,
            str(execution_log),
            job["job_id"],
        ]

    _, spool, ledger, pool = components(tmp_path, argv_hook=hook)
    first = create_predict_job(ledger, spool)
    second = create_predict_job(ledger, spool)
    seal_and_queue(ledger, first, [], device="cuda")
    seal_and_queue(ledger, second, [], device="cuda")

    pool.start()
    try:
        pool.notify_queued()
        deadline = time.time() + 5
        while time.time() < deadline:
            states = [
                ledger.get_job(first["job_id"])["state"],
                ledger.get_job(second["job_id"])["state"],
            ]
            if states == [JobState.SUCCEEDED.value, JobState.SUCCEEDED.value]:
                break
            time.sleep(0.02)
    finally:
        pool.shutdown(timeout=1)

    assert states == [JobState.SUCCEEDED.value, JobState.SUCCEEDED.value]
    assert execution_log.read_text().splitlines() == [
        f"start {first['job_id']}",
        f"end {first['job_id']}",
        f"start {second['job_id']}",
        f"end {second['job_id']}",
    ]
