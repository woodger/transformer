from __future__ import annotations

import uuid

import pytest

from app.contracts.worker.v6.config import ModelConfig, TrainConfig
from app.contracts.worker.v6.objective import (
    TRAINING_RECOVERY_FORMAT,
    ml_contract,
)
from app.service.adapters.inbound.flight.constants import FIT_SCHEMA_ID
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.domain.errors import ServiceError
from app.service.domain.input_manifest import manifest_sha256
from app.service.domain.job import ErrorCode, ExecutionState, InputState

OWNER = "inventory"
DATA_CONTRACT_SHA256 = "d" * 64
SCHEMA_FINGERPRINT = "e" * 64


def _data_contract() -> dict:
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "data_contract_sha256": DATA_CONTRACT_SHA256,
        "seq_len": 2,
        "feature_dim": 2,
        "target_schema_id": "inventory.target.v1",
    }


def _create_fit(ledger: Ledger, *, now: float = 1.0) -> tuple[dict, str]:
    job_id = str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    training_config = TrainConfig(epochs=2, deterministic=True)
    job = ledger.create_job(
        job_id=job_id,
        owner_subject=OWNER,
        client_execution_id=execution_id,
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        data_contract=_data_contract(),
        ml_contract=ml_contract(training_config),
        create_result={"jobId": job_id, "ownership": {"fencingToken": "1"}},
        model_label="daily",
        model_config=ModelConfig(seq_len=2),
        training_config=training_config,
        now=now,
    )
    return job, execution_id


def _reserve_and_commit(
    ledger: Ledger,
    job: dict,
    execution_id: str,
    ordinal: int,
    *,
    rows: int,
    byte_count: int | None = None,
    upload_token: str | None = None,
    payload_id: str | None = None,
    now: float | None = None,
) -> dict:
    upload_token = upload_token or f"upload-{uuid.uuid4()}"
    payload_id = payload_id or str(uuid.uuid4())
    byte_count = rows + 100 if byte_count is None else byte_count
    candidate = f"jobs/{job['job_id']}/inputs/candidates/{upload_token}.arrow"
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=ordinal,
        client_execution_id=execution_id,
        fencing_token=job["fencing_token"],
        upload_token=upload_token,
        candidate_path=candidate,
        storage_class="recovery",
        now=now,
    )
    return ledger.commit_input(
        upload_token=upload_token,
        job_id=job["job_id"],
        client_execution_id=execution_id,
        fencing_token=job["fencing_token"],
        relative_path=candidate,
        schema_id=FIT_SCHEMA_ID,
        data_contract_sha256=DATA_CONTRACT_SHA256,
        rows=rows,
        batches=1,
        byte_count=byte_count,
        sha256=f"{ordinal + 1:064x}",
        schema_fingerprint=SCHEMA_FINGERPRINT,
        source_width=4,
        feature_dim=2,
        selected_device="cpu",
        max_payloads=100,
        max_job_bytes=1_000_000,
        storage_class="recovery",
        now=now,
    )


def _close(ledger: Ledger, job: dict, execution_id: str, *, now=10.0):
    receipts = ledger.list_inputs(job["job_id"])
    return ledger.close_input(
        job["job_id"],
        client_execution_id=execution_id,
        fencing_token=job["fencing_token"],
        payload_count=len(receipts),
        total_rows=sum(item["rows"] for item in receipts),
        total_bytes=sum(item["bytes"] for item in receipts),
        manifest_sha256=manifest_sha256(receipts),
        selected_device="cpu",
        now=now,
    )


def test_input_and_execution_states_are_independent(postgres_ledger):
    job, execution_id = _create_fit(postgres_ledger)

    assert job["input_state"] == InputState.OPEN.value
    assert job["execution_state"] == ExecutionState.WAITING_INPUT.value

    committed = _reserve_and_commit(
        postgres_ledger,
        job,
        execution_id,
        0,
        rows=2,
    )
    queued = postgres_ledger.get_job(job["job_id"])

    assert committed["queued"] is True
    assert committed["next_input_ordinal"] == 1
    assert queued["input_state"] == InputState.OPEN.value
    assert queued["execution_state"] == ExecutionState.QUEUED.value

    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")
    assert running.input_state is InputState.OPEN
    assert running.execution_state is ExecutionState.RUNNING

    closed, repeated = _close(postgres_ledger, job, execution_id)
    assert repeated is False
    assert closed["input_state"] == InputState.CLOSED.value
    assert closed["execution_state"] == ExecutionState.RUNNING.value


def test_nonretryable_worker_failure_aborts_open_input(postgres_ledger):
    job, execution_id = _create_fit(postgres_ledger)
    _reserve_and_commit(
        postgres_ledger,
        job,
        execution_id,
        0,
        rows=2,
    )
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")

    failed = postgres_ledger.finish_attempt(
        job["job_id"],
        running.attempt,
        ExecutionState.FAILED,
        attempt_id=running.attempt_id,
        error_code=ErrorCode.MALFORMED_OUTPUT,
        error_message="worker output is invalid",
    )

    assert failed["execution_state"] == ExecutionState.FAILED.value
    assert failed["input_state"] == InputState.ABORTED.value


def test_status_counts_process_failure_that_restarts_open_fit(postgres_ledger):
    job, execution_id = _create_fit(postgres_ledger)
    _reserve_and_commit(
        postgres_ledger,
        job,
        execution_id,
        0,
        rows=2,
    )
    running = postgres_ledger.claim_execution_job(job["job_id"], "cpu")
    assert postgres_ledger.mark_input_waiting(
        job["job_id"],
        running.attempt,
        attempt_id=running.attempt_id,
        next_ordinal=1,
        input_revision=1,
    )

    retried = postgres_ledger.schedule_retry(
        job["job_id"],
        running.attempt,
        attempt_id=running.attempt_id,
        error_code=ErrorCode.SUBPROCESS_FAILED,
        error_message="worker crashed before EOF",
    )
    snapshot = postgres_ledger.get_status_snapshot_record(job["job_id"], OWNER)

    assert retried["execution_state"] == ExecutionState.RETRYING.value
    assert retried["waiting_for_input"] is False
    assert retried["input_waiting_since"] is None
    assert snapshot.recovery.retry_count == 1
    assert snapshot.recovery.last_retry_code == ErrorCode.SUBPROCESS_FAILED.value


def test_fit_run_summary_counts_recovery_intervals_once(postgres_ledger):
    job, execution_id = _create_fit(postgres_ledger, now=1.0)
    _reserve_and_commit(
        postgres_ledger,
        job,
        execution_id,
        0,
        rows=2,
        byte_count=102,
        now=2.0,
    )
    _close(postgres_ledger, job, execution_id, now=3.0)

    first_attempt = postgres_ledger.claim_execution_job(
        job["job_id"],
        "cpu",
        now=4.0,
    )
    postgres_ledger.mark_attempt_worker_ready(
        job["job_id"],
        first_attempt.attempt,
        attempt_id=first_attempt.attempt_id,
        now=5.0,
    )
    postgres_ledger.register_recovery_checkpoint(
        job_id=job["job_id"],
        attempt=first_attempt.attempt,
        attempt_id=first_attempt.attempt_id,
        generation=1,
        format=TRAINING_RECOVERY_FORMAT,
        relative_path=f"jobs/{job['job_id']}/recovery/1.pth",
        byte_count=10,
        sha256="1" * 64,
        completed_epochs=1,
        global_step=10,
        training_complete=False,
        metrics={"epoch": 1, "step": 10, "elapsed_ms": 100.0},
        checkpoint_serialization_ms=11.0,
        checkpoint_publication_ms=12.0,
        now=6.0,
    )
    postgres_ledger.schedule_retry(
        job["job_id"],
        first_attempt.attempt,
        attempt_id=first_attempt.attempt_id,
        error_code=ErrorCode.SUBPROCESS_FAILED,
        error_message="worker crashed after the first epoch",
        now=7.0,
    )

    second_attempt = postgres_ledger.claim_execution_job(
        job["job_id"],
        "cpu",
        now=8.0,
    )
    postgres_ledger.mark_attempt_worker_ready(
        job["job_id"],
        second_attempt.attempt,
        attempt_id=second_attempt.attempt_id,
        now=9.0,
    )
    postgres_ledger.register_recovery_checkpoint(
        job_id=job["job_id"],
        attempt=second_attempt.attempt,
        attempt_id=second_attempt.attempt_id,
        generation=2,
        format=TRAINING_RECOVERY_FORMAT,
        relative_path=f"jobs/{job['job_id']}/recovery/2.pth",
        byte_count=11,
        sha256="2" * 64,
        completed_epochs=2,
        global_step=20,
        training_complete=True,
        metrics={"epoch": 2, "step": 20, "elapsed_ms": 200.0},
        checkpoint_serialization_ms=13.0,
        checkpoint_publication_ms=14.0,
        now=10.0,
    )
    postgres_ledger.mark_attempt_worker_completed(
        job["job_id"],
        second_attempt.attempt,
        attempt_id=second_attempt.attempt_id,
        now=11.0,
    )

    summary = postgres_ledger.fit_run_summary_source(
        job["job_id"],
        second_attempt.attempt,
        attempt_id=second_attempt.attempt_id,
        now=12.0,
    )

    assert summary.attempt_count == 2
    assert summary.recovery_count == 1
    assert summary.input_payload_count == 1
    assert summary.input_rows == 2
    assert summary.input_bytes == 102
    assert summary.queue_wait_ms == 3_000.0
    assert summary.worker_startup_ms == 2_000.0
    assert summary.training_ms == 300.0
    assert summary.checkpoint_serialization_ms == 24.0
    assert summary.checkpoint_publication_ms == 26.0


def test_out_of_order_commits_advance_only_the_contiguous_frontier(
    postgres_ledger,
):
    job, execution_id = _create_fit(postgres_ledger)

    second = _reserve_and_commit(
        postgres_ledger,
        job,
        execution_id,
        1,
        rows=2,
    )
    assert second["commit_revision"] == 1
    assert second["next_input_ordinal"] == 0
    assert second["queued"] is False

    first = _reserve_and_commit(
        postgres_ledger,
        job,
        execution_id,
        0,
        rows=1,
    )
    assert first["commit_revision"] == 2
    assert first["next_input_ordinal"] == 2
    assert first["queued"] is True
    assert [item.ordinal for item in postgres_ledger.list_committed_inputs(
        job["job_id"]
    )] == [0, 1]


def test_revision_pagination_has_a_stable_snapshot_and_keeps_late_low_ordinal(
    postgres_ledger,
):
    job, execution_id = _create_fit(postgres_ledger)
    _reserve_and_commit(postgres_ledger, job, execution_id, 2, rows=1)
    _reserve_and_commit(postgres_ledger, job, execution_id, 0, rows=1)

    first_page = postgres_ledger.list_inputs_page(
        job["job_id"],
        OWNER,
        after_revision=0,
        snapshot_revision=None,
        cursor=None,
        limit=1,
    )
    assert first_page["snapshot_revision"] == 2
    assert [item["ordinal"] for item in first_page["items"]] == [2]
    assert first_page["next_cursor"] == 1

    _reserve_and_commit(postgres_ledger, job, execution_id, 1, rows=1)
    second_page = postgres_ledger.list_inputs_page(
        job["job_id"],
        OWNER,
        after_revision=0,
        snapshot_revision=first_page["snapshot_revision"],
        cursor=first_page["next_cursor"],
        limit=1,
    )
    assert [item["ordinal"] for item in second_page["items"]] == [0]
    assert second_page["has_more"] is False

    next_cycle = postgres_ledger.list_inputs_page(
        job["job_id"],
        OWNER,
        after_revision=first_page["snapshot_revision"],
        snapshot_revision=None,
        cursor=None,
        limit=100,
    )
    assert [item["ordinal"] for item in next_cycle["items"]] == [1]
    assert next_cycle["snapshot_revision"] == 3


def test_takeover_fences_a_reserved_doput_close_and_cancel(postgres_ledger):
    job, old_execution_id = _create_fit(postgres_ledger)
    payload_id = str(uuid.uuid4())
    postgres_ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=0,
        client_execution_id=old_execution_id,
        fencing_token=1,
        upload_token="late-upload",
        candidate_path=(
            f"jobs/{job['job_id']}/inputs/candidates/late-upload.arrow"
        ),
        storage_class="recovery",
    )

    new_execution_id = str(uuid.uuid4())
    acquired, cleanup = postgres_ledger.acquire_job(
        job["job_id"],
        owner_subject=OWNER,
        previous_client_execution_id=old_execution_id,
        expected_fencing_token=1,
        client_execution_id=new_execution_id,
        acquire_grace_seconds=30,
    )
    assert acquired["fencing_token"] == 2
    assert cleanup == ((
        "recovery",
        f"jobs/{job['job_id']}/inputs/candidates/late-upload.arrow",
    ),)

    with pytest.raises(ServiceError) as late_put:
        postgres_ledger.commit_input(
            upload_token="late-upload",
            job_id=job["job_id"],
            client_execution_id=old_execution_id,
            fencing_token=1,
            relative_path=(
                f"jobs/{job['job_id']}/inputs/candidates/late-upload.arrow"
            ),
            schema_id=FIT_SCHEMA_ID,
            data_contract_sha256=DATA_CONTRACT_SHA256,
            rows=1,
            batches=1,
            byte_count=101,
            sha256="1" * 64,
            schema_fingerprint=SCHEMA_FINGERPRINT,
            source_width=4,
            feature_dim=2,
            selected_device="cpu",
            max_payloads=100,
            max_job_bytes=1_000_000,
            storage_class="recovery",
        )
    assert late_put.value.code is ErrorCode.STALE_FENCE

    empty_manifest = manifest_sha256([])
    with pytest.raises(ServiceError) as late_close:
        postgres_ledger.close_input(
            job["job_id"],
            client_execution_id=old_execution_id,
            fencing_token=1,
            payload_count=0,
            total_rows=0,
            total_bytes=0,
            manifest_sha256=empty_manifest,
            selected_device="cpu",
        )
    assert late_close.value.code is ErrorCode.STALE_FENCE

    with pytest.raises(ServiceError) as late_cancel:
        postgres_ledger.cancel_job(
            job["job_id"],
            owner_subject=OWNER,
            client_execution_id=old_execution_id,
            fencing_token=1,
        )
    assert late_cancel.value.code is ErrorCode.STALE_FENCE


def test_close_rejects_gaps_and_empty_fit_without_closing_input(postgres_ledger):
    job, execution_id = _create_fit(postgres_ledger)
    _reserve_and_commit(postgres_ledger, job, execution_id, 1, rows=1)
    receipts = postgres_ledger.list_inputs(job["job_id"])

    with pytest.raises(ServiceError, match="contiguous"):
        _close(postgres_ledger, job, execution_id)
    assert postgres_ledger.get_job(job["job_id"])["input_state"] == "OPEN"

    empty, empty_execution_id = _create_fit(postgres_ledger)
    with pytest.raises(ServiceError) as failure:
        postgres_ledger.close_input(
            empty["job_id"],
            client_execution_id=empty_execution_id,
            fencing_token=1,
            payload_count=0,
            total_rows=0,
            total_bytes=0,
            manifest_sha256=manifest_sha256([]),
            selected_device="cpu",
        )
    assert failure.value.code is ErrorCode.EMPTY_INPUT
    assert postgres_ledger.get_job(empty["job_id"])["input_state"] == "OPEN"
    assert receipts[0]["ordinal"] == 1


def test_manifest_digest_ignores_arrival_time_and_commit_order(postgres_ledger):
    first_job, first_execution = _create_fit(postgres_ledger)
    second_job, second_execution = _create_fit(postgres_ledger)
    payloads = [str(uuid.uuid4()), str(uuid.uuid4())]

    _reserve_and_commit(
        postgres_ledger,
        first_job,
        first_execution,
        0,
        rows=2,
        payload_id=payloads[0],
        now=10,
    )
    _reserve_and_commit(
        postgres_ledger,
        first_job,
        first_execution,
        1,
        rows=3,
        payload_id=payloads[1],
        now=20,
    )
    _reserve_and_commit(
        postgres_ledger,
        second_job,
        second_execution,
        1,
        rows=3,
        payload_id=payloads[1],
        now=200,
    )
    _reserve_and_commit(
        postgres_ledger,
        second_job,
        second_execution,
        0,
        rows=2,
        payload_id=payloads[0],
        now=100,
    )

    assert manifest_sha256(postgres_ledger.list_inputs(first_job["job_id"])) == (
        manifest_sha256(postgres_ledger.list_inputs(second_job["job_id"]))
    )


def test_job_identity_survives_runtime_row_retention(postgres_ledger):
    job, execution_id = _create_fit(postgres_ledger, now=1)
    cancelled, _, _ = postgres_ledger.cancel_job(
        job["job_id"],
        owner_subject=OWNER,
        client_execution_id=execution_id,
        fencing_token=1,
        now=2,
    )
    assert cancelled["execution_state"] == ExecutionState.CANCELLED.value

    assert postgres_ledger.delete_terminal_jobs_before(3) == [job["job_id"]]
    assert postgres_ledger.get_job(job["job_id"]) is None
    identity = postgres_ledger.get_job_identity(job["job_id"])
    assert identity["retired_at"] is not None
    assert identity["create_hash"] == "a" * 64
