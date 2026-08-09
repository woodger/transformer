import uuid
from dataclasses import FrozenInstanceError, fields

import pytest

from app.flight.constants import FIT_SCHEMA_ID, JobState
from app.flight.records import (
    CommittedInputRecord,
    ExecutionJobRecord,
    ModelArtifactRecord,
    RecoverableAttemptRecord,
)
from app.training.run_config import ModelConfig, TrainConfig

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _create_fit_job(
    ledger,
    *,
    owner_subject="inventory",
    model_config=None,
    training_config=None,
    now=100.0,
):
    job_id = str(uuid.uuid4())
    ledger.create_job(
        job_id=job_id,
        owner_subject=owner_subject,
        operation="fit",
        requested_device="cpu",
        prediction_column="prediction",
        config_hash=DIGEST_A,
        model_label="returns.daily",
        model_config=model_config,
        training_config=training_config,
        now=now,
    )
    return job_id


def _seal_and_queue(ledger, job_id, *, manifest=None, now=101.0):
    ledger.seal_job(
        job_id,
        manifest_hash=DIGEST_A,
        manifest=[] if manifest is None else manifest,
        source_width=12,
        feature_dim=3,
        now=now,
    )
    ledger.queue_job(
        job_id,
        selected_device="cpu",
        now=now + 1,
    )


def _assert_frozen_slots(record):
    assert not hasattr(record, "__dict__")
    field = fields(record)[0]
    with pytest.raises(FrozenInstanceError):
        setattr(record, field.name, getattr(record, field.name))


def test_execution_job_mapper_preserves_types_timestamps_and_legacy_shape(
    postgres_ledger,
):
    model_config = ModelConfig(
        seq_len=4,
        hidden=16,
        layers=2,
        dropout=0.2,
        nhead=4,
        context_mode="strict",
        feature_dim=3,
    )
    training_config = TrainConfig(
        lr=0.001,
        batch_size=8,
        epochs=3,
        patience=1,
    )
    job_id = _create_fit_job(
        postgres_ledger,
        model_config=model_config,
        training_config=training_config,
    )
    manifest = [
        {"payloadId": str(uuid.uuid4()), "ordinal": ordinal, "sha256": DIGEST_A}
        for ordinal in range(2)
    ]
    _seal_and_queue(postgres_ledger, job_id, manifest=manifest)

    queued = postgres_ledger.queued_execution_jobs()

    assert len(queued) == 1
    queued_job = queued[0]
    assert isinstance(queued_job, ExecutionJobRecord)
    assert queued_job.job_id == job_id
    assert queued_job.state is JobState.QUEUED
    assert queued_job.model_config == model_config
    assert isinstance(queued_job.model_config, ModelConfig)
    assert queued_job.training_config == training_config
    assert isinstance(queued_job.training_config, TrainConfig)
    assert queued_job.feature_dim == 3
    assert queued_job.input_frame_count == 2
    assert queued_job.attempt == 0
    assert queued_job.queued_at == 102.0
    assert queued_job.started_at is None
    _assert_frozen_slots(queued_job)

    running = postgres_ledger.claim_execution_job(
        job_id,
        "cpu",
        worker_id="records-test",
        now=103.0,
    )

    assert isinstance(running, ExecutionJobRecord)
    assert running.state is JobState.RUNNING
    assert running.attempt == 1
    assert running.queued_at == 102.0
    assert running.started_at == 103.0
    assert postgres_ledger.get_execution_job(job_id) == running

    legacy = postgres_ledger.get_job(job_id)
    assert legacy["state"] == "RUNNING"
    assert isinstance(legacy["state"], str)
    assert legacy["model_config"] == model_config.to_dict()
    assert legacy["training_config"] == training_config.to_dict()
    assert legacy["queued_at"] == 102.0
    assert legacy["started_at"] == 103.0


def test_committed_input_mapper_renames_bytes_without_changing_legacy_shape(
    postgres_ledger,
):
    job_id = _create_fit_job(postgres_ledger)
    payload_id = str(uuid.uuid4())
    postgres_ledger.reserve_input(
        job_id=job_id,
        payload_id=payload_id,
        ordinal=0,
        upload_token="records-input",
        temporary_path=f"spool/jobs/{job_id}/inputs/.0.tmp",
    )
    byte_count = 4 * 1024 * 1024 + 17
    postgres_ledger.commit_input(
        upload_token="records-input",
        relative_path=f"spool/jobs/{job_id}/inputs/0.arrow",
        schema_id=FIT_SCHEMA_ID,
        rows=255,
        batches=4,
        byte_count=byte_count,
        sha256=DIGEST_A,
        schema_fingerprint=DIGEST_B,
        source_width=12,
        feature_dim=3,
        max_payloads=4,
        max_job_bytes=8 * 1024 * 1024,
    )

    records = postgres_ledger.list_committed_inputs(job_id)

    assert records == [
        CommittedInputRecord(
            job_id=job_id,
            ordinal=0,
            schema_id=FIT_SCHEMA_ID,
            rows=255,
            byte_count=byte_count,
            sha256=DIGEST_A,
            relative_path=f"spool/jobs/{job_id}/inputs/0.arrow",
            storage_class="runtime",
        )
    ]
    _assert_frozen_slots(records[0])

    legacy = postgres_ledger.list_inputs(job_id)
    assert legacy[0]["bytes"] == byte_count
    assert "byte_count" not in legacy[0]


def test_model_artifact_mapper_enforces_owner_and_preserves_legacy_record(
    postgres_ledger,
):
    owner_subject = "inventory-a"
    job_id = _create_fit_job(
        postgres_ledger,
        owner_subject=owner_subject,
    )
    _seal_and_queue(postgres_ledger, job_id)
    running = postgres_ledger.claim_execution_job(
        job_id,
        "cpu",
        worker_id="records-model",
    )
    model_ref = f"mdl_{uuid.uuid4().hex}"
    checkpoint_path = f"models/{model_ref}/checkpoint.pth"
    postgres_ledger.publish_model(
        job_id,
        running.attempt,
        attempt_id=running.attempt_id,
        model_ref=model_ref,
        label="returns.daily",
        generation=None,
        checkpoint_path=checkpoint_path,
        metadata_path=f"models/{model_ref}/metadata.json",
        sha256=DIGEST_B,
        metadata={"modelRef": model_ref},
        result={"modelRef": model_ref},
    )

    artifact = postgres_ledger.get_model_artifact(
        model_ref,
        owner_subject=owner_subject,
    )

    assert artifact == ModelArtifactRecord(
        model_ref=model_ref,
        owner_subject=owner_subject,
        checkpoint_path=checkpoint_path,
        sha256=DIGEST_B,
    )
    _assert_frozen_slots(artifact)
    assert postgres_ledger.get_model_artifact(
        model_ref,
        owner_subject="inventory-b",
    ) is None

    legacy = postgres_ledger.get_model(
        model_ref,
        owner_subject=owner_subject,
    )
    assert legacy["model_ref"] == model_ref
    assert legacy["owner_subject"] == owner_subject
    assert legacy["checkpoint_path"] == checkpoint_path
    assert legacy["sha256"] == DIGEST_B


def test_recoverable_attempt_mapper_preserves_process_identity_and_legacy_shape(
    postgres_ledger,
):
    job_id = _create_fit_job(postgres_ledger)
    _seal_and_queue(postgres_ledger, job_id)
    running = postgres_ledger.claim_execution_job(
        job_id,
        "cpu",
        worker_id="records-recovery",
    )
    boot_id = str(uuid.uuid4())
    postgres_ledger.set_attempt_process(
        job_id,
        running.attempt,
        attempt_id=running.attempt_id,
        pid=12001,
        pgid=12002,
        boot_id=boot_id,
        process_start_ticks=987654,
    )

    attempts = postgres_ledger.list_recoverable_attempts()

    assert attempts == [
        RecoverableAttemptRecord(
            job_id=job_id,
            attempt=1,
            pid=12001,
            pgid=12002,
            boot_id=boot_id,
            process_start_ticks=987654,
            attempt_id=running.attempt_id,
        )
    ]
    _assert_frozen_slots(attempts[0])

    legacy = postgres_ledger.list_active_attempts()
    assert legacy[0]["job_id"] == job_id
    assert legacy[0]["attempt"] == 1
    assert legacy[0]["pid"] == 12001
    assert legacy[0]["pgid"] == 12002
    assert legacy[0]["boot_id"] == boot_id
    assert legacy[0]["process_start_ticks"] == 987654
