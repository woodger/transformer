from __future__ import annotations

import hashlib
import uuid

from app.contracts.worker.v3.config import ModelConfig, TrainConfig
from app.contracts.worker.v3.objective import ml_contract
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
