from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

from app.contracts.metrics.fit_run.v2 import build_run_summary
from app.contracts.metrics.v2 import (
    build_training_record,
)
from app.contracts.worker.v6.config import ModelConfig, TrainConfig
from app.contracts.worker.v6.objective import (
    CHECKPOINT_FORMAT,
    ml_contract,
    objective_config_sha256,
)
from app.service.adapters.inbound.flight.constants import (
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
)
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.cuda.inventory import (
    static_cuda_inventory,
)
from app.service.bootstrap.control_plane import build_job_coordinator
from app.service.domain.input_manifest import manifest_sha256

OWNER = "inventory"
DATA_CONTRACT_SHA256 = "d" * 64
SCHEMA_FINGERPRINT = "e" * 64


@dataclass(frozen=True, slots=True)
class TestMetricsArtifact:
    relative_path: str
    byte_count: int
    sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class TestRunSummaryArtifact:
    relative_path: str
    byte_count: int
    sha256: str


def build_test_job_coordinator(
    config,
    ledger,
    spool,
    *,
    cuda_available=None,
    device_inventory=None,
    recovery_store=None,
    metrics=None,
    logger=None,
    cancel_notifier=None,
    queue_notifier=None,
):
    """Assemble a coordinator with deterministic test dependencies."""
    inventory = device_inventory
    if inventory is None:
        inventory = static_cuda_inventory(
            cuda_available or (lambda: False),
        )
    return build_job_coordinator(
        config,
        ledger,
        spool,
        recovery_store or spool,
        device_inventory=inventory,
        metrics=metrics or OperationalMetrics(),
        logger=logger or JsonLogger(),
        cancel_notifier=cancel_notifier,
        queue_notifier=queue_notifier,
    )


def internal_data_contract(*, digest=DATA_CONTRACT_SHA256):
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "data_contract_sha256": digest,
        "seq_len": 2,
        "feature_dim": 2,
        "target_schema_id": "inventory.target.v1",
    }


def public_data_contract(*, digest=DATA_CONTRACT_SHA256):
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "dataContractSha256": digest,
        "seqLen": 2,
        "featureDim": 2,
        "targetSchemaId": "inventory.target.v1",
    }


def model_config():
    return ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        feature_dim=2,
    )


def train_config():
    return TrainConfig(
        batch_size=2,
        epochs=2,
        loss_schedule="none",
        use_amp=False,
        deterministic=True,
        seed=17,
    )


def public_ml_contract():
    return ml_contract(train_config())


def create_test_metrics_artifact(
    spool,
    *,
    model_ref,
    job_id,
    attempt_id,
    attempt,
):
    metrics = {
        "mode": "fit-stream",
        "frame": None,
        "epoch": 1,
        "step": 1,
        "rows": 1,
        "batches": 1,
        "lr": 0.001,
        "loss_stage": 4,
        "minimum_loss_stage": 4,
        "maximum_loss_stage": 4,
        "loss": 0.5,
        "loss_l0": 0.1,
        "loss_l1": 0.1,
        "loss_l2": 0.1,
        "loss_l3": 0.1,
        "loss_l4": 0.1,
        "loss_l5": 0.1,
        "loss_nll": 0.0,
        "loss_ev": 0.0,
        "mean_return_mae": 0.1,
        "sigma_return_mae": 0.1,
        "prob_tp_mae": 0.1,
        "prob_sl_mae": 0.1,
        "volatility_next_mae": 0.1,
        "hitting_prob_tp_mae": 0.1,
        "mean_return_rmse": 0.1,
        "sigma_return_rmse": 0.1,
        "prob_tp_rmse": 0.1,
        "prob_sl_rmse": 0.1,
        "volatility_next_rmse": 0.1,
        "hitting_prob_tp_rmse": 0.1,
        "selection_score": None,
        "trainingBatchesCompleted": 1,
        "optimizerUpdatesApplied": 1,
        "optimizerUpdatesSkipped": 0,
        "ampOverflowBatches": 0,
        "finiteGradientBatches": 1,
        "nonFiniteGradientBatches": 0,
        "preClipGradientNormMean": 1.0,
        "preClipGradientNormMax": 1.0,
        "preClipGradientNormP95": 1.0,
        "nan_ratio": 0.0,
        "masked_token_ratio": 0.0,
        "complete_token_ratio": 1.0,
        "partial_token_ratio": 0.0,
        "empty_token_ratio": 0.0,
        "input_pipeline_ms": 1.0,
        "missing_stats_ms": 1.0,
        "host_to_device_ms": 1.0,
        "train_step_ms": 1.0,
        "elapsed_ms": 4.0,
        "checkpoint_best": False,
        "should_stop": False,
        "best_selection_score": None,
    }
    record = build_training_record(
        metrics,
        recorded_at=1.0,
        job_id=job_id,
        attempt_id=attempt_id,
        attempt=attempt,
        model_ref=model_ref,
        data_contract_sha256=DATA_CONTRACT_SHA256,
        objective_config_sha256=objective_config_sha256(train_config()),
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.10",
        git_commit="0" * 40,
    )
    payload = (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path = spool.telemetry_metrics_path(job_id)
    spool.atomic_write_bytes(path, payload)
    return TestMetricsArtifact(
        relative_path=spool.telemetry_relative_path(path),
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        row_count=1,
    )


def create_test_run_summary_artifact(
    spool,
    *,
    model_ref,
    job_id,
    attempt_id,
    attempt,
):
    summary = build_run_summary(
        recorded_at=10.0,
        job_id=job_id,
        attempt_id=attempt_id,
        attempt=attempt,
        model_ref=model_ref,
        data_contract_sha256=DATA_CONTRACT_SHA256,
        objective_config_sha256=objective_config_sha256(train_config()),
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.10",
        git_commit="0" * 40,
        milestones={
            "createdAt": "1970-01-01T00:00:01.000Z",
            "firstInputCommittedAt": "1970-01-01T00:00:02.000Z",
            "inputClosedAt": "1970-01-01T00:00:03.000Z",
            "workerCompletedAt": "1970-01-01T00:00:09.000Z",
            "publishedAt": "1970-01-01T00:00:10.000Z",
        },
        durations={
            "firstInputWaitMs": 1000.0,
            "eofWaitMs": 1000.0,
            "queueWaitMs": 1000.0,
            "workerStartupMs": 1000.0,
            "trainingMs": 4000.0,
            "checkpointSerializationMs": 1.0,
            "checkpointPublicationMs": 1.0,
            "modelPublicationMs": 1000.0,
            "remoteFitMs": 9000.0,
        },
        counts={
            "attempts": attempt,
            "recoveries": max(0, attempt - 1),
            "inputPayloads": 1,
            "inputRows": 1,
            "inputBytes": 1,
        },
    )
    payload = json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path = spool.telemetry_run_summary_path(job_id)
    spool.atomic_write_bytes(path, payload)
    return TestRunSummaryArtifact(
        relative_path=spool.telemetry_relative_path(path),
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def create_fit(ledger, *, job_id=None, execution_id=None, now=None):
    job_id = job_id or str(uuid.uuid4())
    execution_id = execution_id or str(uuid.uuid4())
    job = ledger.create_job(
        job_id=job_id,
        owner_subject=OWNER,
        client_execution_id=execution_id,
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        data_contract=internal_data_contract(),
        ml_contract=public_ml_contract(),
        create_result={"jobId": job_id},
        model_label="daily",
        model_config=model_config(),
        training_config=train_config(),
        now=now,
    )
    return job


def create_predict(
    ledger,
    *,
    model_ref="mdl_seed",
    job_id=None,
    execution_id=None,
    now=None,
):
    job_id = job_id or str(uuid.uuid4())
    execution_id = execution_id or str(uuid.uuid4())
    return ledger.create_job(
        job_id=job_id,
        owner_subject=OWNER,
        client_execution_id=execution_id,
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="b" * 64,
        data_contract=internal_data_contract(),
        ml_contract=public_ml_contract(),
        create_result={"jobId": job_id},
        resolved_model_ref=model_ref,
        model_config=model_config(),
        now=now,
    )


def commit_input(
    ledger,
    job,
    ordinal,
    *,
    rows=1,
    storage_class="recovery",
    payload_id=None,
    selected_device="cpu",
    now=None,
):
    payload_id = payload_id or str(uuid.uuid4())
    upload_token = uuid.uuid4().hex
    relative_path = (
        f"jobs/{job['job_id']}/inputs/"
        f"{ordinal}-{payload_id}-{upload_token}.arrow"
    )
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=ordinal,
        client_execution_id=job["client_execution_id"],
        fencing_token=job["fencing_token"],
        upload_token=upload_token,
        candidate_path=relative_path,
        storage_class=storage_class,
        now=now,
    )
    return ledger.commit_input(
        upload_token=upload_token,
        job_id=job["job_id"],
        client_execution_id=job["client_execution_id"],
        fencing_token=job["fencing_token"],
        relative_path=relative_path,
        schema_id=(
            FIT_SCHEMA_ID
            if job["operation"] == "fit"
            else PREDICT_SCHEMA_ID
        ),
        data_contract_sha256=DATA_CONTRACT_SHA256,
        rows=rows,
        batches=1,
        byte_count=100 + rows,
        sha256=hashlib.sha256(
            f"{job['job_id']}:{ordinal}".encode()
        ).hexdigest(),
        schema_fingerprint=SCHEMA_FINGERPRINT,
        source_width=4,
        feature_dim=2,
        selected_device=selected_device,
        max_payloads=100,
        max_job_bytes=1_000_000,
        storage_class=storage_class,
        now=now,
    )


def close_input(ledger, job, *, selected_device="cpu", now=None):
    current = ledger.get_job(job["job_id"])
    receipts = ledger.list_inputs(job["job_id"])
    return ledger.close_input(
        job["job_id"],
        client_execution_id=current["client_execution_id"],
        fencing_token=current["fencing_token"],
        payload_count=len(receipts),
        total_rows=sum(item["rows"] for item in receipts),
        total_bytes=sum(item["bytes"] for item in receipts),
        manifest_sha256=manifest_sha256(receipts),
        selected_device=selected_device,
        now=now,
    )
