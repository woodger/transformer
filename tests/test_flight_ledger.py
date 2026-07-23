import threading
import uuid

import pytest

from app.database.models import JobAttempt
from app.database.session import Database
from app.flight.constants import ErrorCode, JobState
from app.flight.errors import ServiceError
from app.flight.ledger import Ledger

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


@pytest.fixture
def ledger(postgres_ledger):
    return postgres_ledger


def create_job(ledger, *, operation="fit", owner="inventory", now=1.0):
    job_id = str(uuid.uuid4())
    arguments = {
        "job_id": job_id,
        "owner_subject": owner,
        "operation": operation,
        "requested_device": "cpu",
        "prediction_column": "out",
        "config_hash": DIGEST_A,
        "now": now,
    }
    if operation == "fit":
        arguments["model_label"] = "daily-model"
    else:
        arguments["input_model_ref"] = "model-generation-1"
    return ledger.create_job(**arguments)


def seal_and_queue(ledger, job_id, *, device="cpu", now=2.0):
    ledger.seal_job(
        job_id,
        manifest_hash=DIGEST_A,
        manifest=[],
        source_width=4,
        feature_dim=2,
        result={"jobId": job_id, "state": "SEALED"},
        now=now,
    )
    ledger.queue_job(
        job_id,
        selected_device=device,
        result={"jobId": job_id, "state": "QUEUED"},
        now=now + 1,
    )


def test_job_transitions_increment_revision_and_terminal_state_is_immutable(ledger):
    job = create_job(ledger)
    assert job["state"] == JobState.UPLOADING.value
    assert job["revision"] == 1

    sealed = ledger.transition_job(job["job_id"], JobState.SEALED, now=2.0)
    assert sealed["revision"] == 2

    cancelled = ledger.transition_job(sealed["job_id"], JobState.CANCELLED, now=3.0)
    assert cancelled["revision"] == 3

    with pytest.raises(ValueError, match="invalid job state transition"):
        ledger.transition_job(cancelled["job_id"], JobState.QUEUED)


def test_idempotent_mutation_is_atomic_and_conflicting_key_is_rejected(ledger):
    job_id = str(uuid.uuid4())
    calls = []

    def mutation(connection):
        calls.append(True)
        ledger.create_job(
            job_id=job_id,
            owner_subject="inventory",
            operation="fit",
            requested_device="cpu",
            prediction_column="out",
            config_hash=DIGEST_A,
            model_label="daily-model",
            connection=connection,
        )
        return {"jobId": job_id, "revision": 1}, job_id

    first, replayed = ledger.run_idempotent(
        owner_subject="inventory",
        action_name="transformer.v1.job.create",
        idempotency_key="create-1",
        request_hash=DIGEST_A,
        mutation=mutation,
    )
    repeated, repeated_replayed = ledger.run_idempotent(
        owner_subject="inventory",
        action_name="transformer.v1.job.create",
        idempotency_key="create-1",
        request_hash=DIGEST_A,
        mutation=mutation,
    )

    assert first == repeated == {"jobId": job_id, "revision": 1}
    assert replayed is False
    assert repeated_replayed is True
    assert len(calls) == 1

    with pytest.raises(ServiceError) as error:
        ledger.run_idempotent(
            owner_subject="inventory",
            action_name="transformer.v1.job.create",
            idempotency_key="create-1",
            request_hash=DIGEST_B,
            mutation=mutation,
        )
    assert error.value.code == ErrorCode.ALREADY_EXISTS


def test_idempotent_mutation_and_replay_record_roll_back_together(ledger):
    job_id = str(uuid.uuid4())

    def failing_mutation(connection):
        ledger.create_job(
            job_id=job_id,
            owner_subject="inventory",
            operation="fit",
            requested_device="cpu",
            prediction_column="out",
            config_hash=DIGEST_A,
            model_label="daily-model",
            connection=connection,
        )
        raise RuntimeError("injected mutation failure")

    with pytest.raises(RuntimeError, match="injected mutation failure"):
        ledger.run_idempotent(
            owner_subject="inventory",
            action_name="transformer.v1.job.create",
            idempotency_key="create-rollback",
            request_hash=DIGEST_A,
            mutation=failing_mutation,
        )

    assert ledger.get_job(job_id) is None
    assert ledger.lookup_idempotency(
        owner_subject="inventory",
        action_name="transformer.v1.job.create",
        idempotency_key="create-rollback",
    ) is None


def test_committed_input_is_durable_unique_and_increments_job_revision(
    ledger,
    postgres_config,
):
    job = create_job(ledger)
    payload_id = str(uuid.uuid4())
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=0,
        upload_token="upload-1",
        temporary_path=f"spool/jobs/{job['job_id']}/inputs/.0.tmp",
    )

    committed = ledger.commit_input(
        upload_token="upload-1",
        relative_path=f"spool/jobs/{job['job_id']}/inputs/0.arrow",
        schema_id="inventory.sequence.fit.v1",
        rows=255,
        batches=4,
        byte_count=5 * 1024 * 1024,
        sha256=DIGEST_A,
        schema_fingerprint=DIGEST_B,
        source_width=4,
        feature_dim=2,
        max_payloads=4,
        max_job_bytes=10 * 1024 * 1024,
    )

    assert committed["ordinal"] == 0
    assert committed["batches"] == 4
    assert committed["revision"] == 2
    reopened = Ledger(Database(postgres_config)).initialize()
    try:
        assert reopened.list_inputs(job["job_id"]) == [
            {key: value for key, value in committed.items() if key != "revision"}
        ]
    finally:
        reopened.close()

    with pytest.raises(ServiceError) as error:
        ledger.reserve_input(
            job_id=job["job_id"],
            payload_id=str(uuid.uuid4()),
            ordinal=0,
            upload_token="upload-2",
            temporary_path=f"spool/jobs/{job['job_id']}/inputs/.0.retry.tmp",
        )
    assert error.value.code == ErrorCode.ALREADY_EXISTS


def test_input_quota_failure_does_not_create_committed_input(ledger):
    job = create_job(ledger)
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=str(uuid.uuid4()),
        ordinal=0,
        upload_token="upload-quota",
        temporary_path=f"spool/jobs/{job['job_id']}/inputs/.0.tmp",
    )

    with pytest.raises(ServiceError) as error:
        ledger.commit_input(
            upload_token="upload-quota",
            relative_path=f"spool/jobs/{job['job_id']}/inputs/0.arrow",
            schema_id="inventory.sequence.fit.v1",
            rows=1,
            batches=1,
            byte_count=11,
            sha256=DIGEST_A,
            schema_fingerprint=DIGEST_B,
            source_width=4,
            feature_dim=2,
            max_payloads=1,
            max_job_bytes=10,
        )

    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert ledger.list_inputs(job["job_id"]) == []
    assert ledger.abort_input("upload-quota") is not None


def test_claim_is_atomic_across_worker_threads(ledger):
    first = create_job(ledger, now=1.0)
    second = create_job(ledger, now=2.0)
    seal_and_queue(ledger, first["job_id"], now=3.0)
    seal_and_queue(ledger, second["job_id"], now=4.0)
    barrier = threading.Barrier(3)
    claimed = []
    errors = []

    def claim(worker_id):
        try:
            barrier.wait(timeout=5)
            claimed.append(ledger.claim_next_job("cpu", worker_id=worker_id))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=5)
    finally:
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert {job["job_id"] for job in claimed} == {first["job_id"], second["job_id"]}
    assert all(job["state"] == JobState.RUNNING.value for job in claimed)
    assert all(job["attempt"] == 1 for job in claimed)


def test_claim_fifo_uses_durable_queue_sequence_when_timestamps_tie(ledger):
    first = ledger.create_job(
        job_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
        owner_subject="inventory",
        operation="fit",
        requested_device="cuda",
        prediction_column="out",
        config_hash=DIGEST_A,
        model_label="first",
        now=10.0,
    )
    second = ledger.create_job(
        job_id="00000000-0000-4000-8000-000000000000",
        owner_subject="inventory",
        operation="fit",
        requested_device="cuda",
        prediction_column="out",
        config_hash=DIGEST_B,
        model_label="second",
        now=10.0,
    )
    seal_and_queue(ledger, first["job_id"], device="cuda", now=10.0)
    seal_and_queue(ledger, second["job_id"], device="cuda", now=10.0)

    first_claim = ledger.claim_next_job("cuda", now=11.0)
    second_claim = ledger.claim_next_job("cuda", now=11.0)

    assert first_claim["job_id"] == first["job_id"]
    assert second_claim["job_id"] == second["job_id"]
    assert first_claim["queue_sequence"] < second_claim["queue_sequence"]


def test_output_publication_is_all_or_nothing(ledger):
    job = create_job(ledger, operation="predict")
    seal_and_queue(ledger, job["job_id"])
    running = ledger.claim_next_job("cpu", worker_id="worker")
    output = {
        "ordinal": 0,
        "rows": 1,
        "batches": 1,
        "bytes": 100,
        "sha256": DIGEST_A,
        "schema_fingerprint": DIGEST_B,
        "relative_path": f"spool/jobs/{job['job_id']}/attempts/1/outputs/0.arrow",
    }

    with pytest.raises(ServiceError) as error:
        ledger.publish_outputs(
            job["job_id"],
            running["attempt"],
            [output, output],
            result={"outputs": [0]},
        )
    assert error.value.code == ErrorCode.ALREADY_EXISTS
    assert ledger.list_outputs(job["job_id"]) == []
    assert ledger.get_job(job["job_id"])["state"] == JobState.RUNNING.value

    succeeded = ledger.publish_outputs(
        job["job_id"],
        running["attempt"],
        [output],
        result={"outputs": [0]},
    )
    assert succeeded["state"] == JobState.SUCCEEDED.value
    assert len(ledger.list_outputs(job["job_id"])) == 1


def test_failed_attempt_updates_job_and_attempt_in_one_transaction(ledger):
    job = create_job(ledger)
    seal_and_queue(ledger, job["job_id"])
    running = ledger.claim_next_job("cpu")

    failed = ledger.finish_attempt(
        job["job_id"],
        running["attempt"],
        JobState.FAILED,
        error_code=ErrorCode.SUBPROCESS_FAILED,
        error_message="worker exited with status 2",
        exit_code=2,
        now=10.0,
    )

    assert failed["state"] == JobState.FAILED.value
    assert failed["error_code"] == ErrorCode.SUBPROCESS_FAILED.value
    with ledger.connection() as connection:
        attempt = connection.get(JobAttempt, (job["job_id"], 1))
    assert attempt.status == JobState.FAILED.value
    assert attempt.exit_code == 2


def test_cancelled_attempt_cannot_persist_error_metadata(ledger):
    job = create_job(ledger)
    seal_and_queue(ledger, job["job_id"])
    running = ledger.claim_next_job("cpu")
    ledger.transition_job(job["job_id"], JobState.CANCELLING)

    with pytest.raises(ValueError, match="must not carry an error"):
        ledger.finish_attempt(
            job["job_id"],
            running["attempt"],
            JobState.CANCELLED,
            error_code=ErrorCode.CANCELLED,
            error_message="job was cancelled",
        )

    cancelled = ledger.finish_attempt(
        job["job_id"],
        running["attempt"],
        JobState.CANCELLED,
    )
    assert cancelled["error_code"] is None
    assert cancelled["error_message"] is None


def test_ticket_is_opaque_owner_bound_and_expires(ledger):
    job = create_job(ledger, operation="predict", owner="inventory-a")
    seal_and_queue(ledger, job["job_id"])
    running = ledger.claim_next_job("cpu")
    ledger.publish_outputs(
        job["job_id"],
        running["attempt"],
        [{
            "ordinal": 0,
            "rows": 0,
            "batches": 0,
            "bytes": 128,
            "sha256": DIGEST_A,
            "schema_fingerprint": DIGEST_B,
            "relative_path": f"spool/jobs/{job['job_id']}/attempts/1/outputs/0.arrow",
        }],
        result={"outputs": [0]},
        now=10.0,
    )
    ticket, expires_at = ledger.issue_ticket(
        job_id=job["job_id"],
        ordinal=0,
        owner_subject="inventory-a",
        ttl_seconds=60,
        now=20.0,
    )

    assert b"spool" not in ticket
    assert ledger.resolve_ticket(ticket, owner_subject="inventory-a", now=79.0)["ordinal"] == 0
    with pytest.raises(ServiceError) as owner_error:
        ledger.resolve_ticket(ticket, owner_subject="inventory-b", now=30.0)
    assert owner_error.value.code == ErrorCode.PERMISSION_DENIED
    with pytest.raises(ServiceError) as expiry_error:
        ledger.resolve_ticket(ticket, owner_subject="inventory-a", now=expires_at)
    assert expiry_error.value.code == ErrorCode.FAILED_PRECONDITION


def test_retention_keeps_recent_idempotency_then_deletes_it_with_job(ledger):
    job = create_job(ledger, now=1.0)
    ledger.transition_job(
        job["job_id"],
        JobState.CANCELLED,
        updates={"finished_at": 10.0},
        now=10.0,
    )
    ledger.record_idempotency(
        owner_subject="inventory",
        action_name="transformer.v1.job.cancel",
        idempotency_key="cancel-retained",
        request_hash=DIGEST_A,
        response={"jobId": job["job_id"], "state": "CANCELLED"},
        job_id=job["job_id"],
        now=80.0,
    )

    assert ledger.delete_terminal_jobs_before(50.0) == []
    assert ledger.get_job(job["job_id"]) is not None
    assert ledger.lookup_idempotency(
        "inventory", "transformer.v1.job.cancel", "cancel-retained"
    ) is not None

    assert ledger.delete_terminal_jobs_before(81.0) == [job["job_id"]]
    assert ledger.get_job(job["job_id"]) is None
    assert ledger.lookup_idempotency(
        "inventory", "transformer.v1.job.cancel", "cancel-retained"
    ) is None


def test_retention_does_not_delete_job_with_active_output_ticket(ledger):
    job = create_job(ledger, operation="predict", now=1.0)
    seal_and_queue(ledger, job["job_id"], now=2.0)
    running = ledger.claim_next_job("cpu", now=3.0)
    ledger.publish_outputs(
        job["job_id"],
        running["attempt"],
        [{
            "ordinal": 0,
            "rows": 0,
            "batches": 0,
            "bytes": 128,
            "sha256": DIGEST_A,
            "schema_fingerprint": DIGEST_B,
            "relative_path": f"spool/jobs/{job['job_id']}/attempts/1/outputs/0.arrow",
        }],
        result={"outputs": [0]},
        now=10.0,
    )
    ticket, _ = ledger.issue_ticket(
        job_id=job["job_id"],
        ordinal=0,
        owner_subject="inventory",
        ttl_seconds=20,
        now=90.0,
    )

    assert ledger.delete_terminal_jobs_before(50.0) == []
    assert ledger.resolve_ticket(
        ticket,
        owner_subject="inventory",
        now=100.0,
    )["job_id"] == job["job_id"]


def test_recovery_preserves_safe_states_and_never_requeues_running_fit(ledger):
    uploading = create_job(ledger)
    ledger.reserve_input(
        job_id=uploading["job_id"],
        payload_id=str(uuid.uuid4()),
        ordinal=0,
        upload_token="stale-upload",
        temporary_path=f"spool/jobs/{uploading['job_id']}/inputs/.0.tmp",
    )
    sealed = create_job(ledger)
    ledger.seal_job(
        sealed["job_id"],
        manifest_hash=DIGEST_A,
        manifest=[],
        now=5.0,
    )
    sealed_revision = ledger.get_job(sealed["job_id"])["revision"]
    to_interrupt = create_job(ledger)
    seal_and_queue(ledger, to_interrupt["job_id"])
    running = ledger.claim_next_job("cpu")

    cancelling = create_job(ledger)
    seal_and_queue(ledger, cancelling["job_id"], now=10.0)
    cancelling_running = ledger.claim_next_job("cpu")
    ledger.transition_job(cancelling_running["job_id"], JobState.CANCELLING)

    queued = create_job(ledger)
    seal_and_queue(ledger, queued["job_id"], now=20.0)

    recovery = ledger.reconcile_interrupted_jobs(now=100.0)

    assert ledger.get_job(uploading["job_id"])["state"] == JobState.UPLOADING.value
    recovered_sealed = ledger.get_job(sealed["job_id"])
    assert recovered_sealed["state"] == JobState.SEALED.value
    assert recovered_sealed["revision"] == sealed_revision
    assert ledger.get_job(queued["job_id"])["state"] == JobState.QUEUED.value
    # No RUNNING fit is ever returned to the queue during recovery.
    assert ledger.get_job(running["job_id"])["state"] == JobState.FAILED.value
    assert ledger.get_job(running["job_id"])["error_code"] == ErrorCode.EXECUTION_INTERRUPTED.value
    assert ledger.get_job(cancelling["job_id"])["state"] == JobState.CANCELLED.value
    assert recovery["interrupted_jobs"] == [running["job_id"]]
    assert recovery["cancelled_jobs"] == [cancelling["job_id"]]
    assert recovery["temporary_paths"] == [
        f"spool/jobs/{uploading['job_id']}/inputs/.0.tmp"
    ]
    assert ledger.abort_input("stale-upload") is None
    with ledger.connection() as connection:
        interrupted_attempt = connection.get(JobAttempt, (running["job_id"], 1))
        cancelled_attempt = connection.get(JobAttempt, (cancelling["job_id"], 1))
    assert interrupted_attempt.status == JobState.FAILED.value
    assert interrupted_attempt.error_code == ErrorCode.EXECUTION_INTERRUPTED.value
    assert cancelled_attempt.status == JobState.CANCELLED.value
