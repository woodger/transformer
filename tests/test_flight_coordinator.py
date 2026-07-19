import json
import uuid

import pytest

from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CANCEL_ACTION,
    CONTRACT_NAME,
    CREATE_ACTION,
    HEALTH_ACTION,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
    JobState,
)
from app.flight.contract import validate_action_request
from app.flight.coordinator import JobCoordinator
from app.flight.errors import ServiceError
from app.flight.ledger import Ledger
from app.flight.spool import Spool


def common():
    return {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": str(uuid.uuid4()),
    }


def create_document(**overrides):
    document = {
        **common(),
        "idempotencyKey": "create-fit-1",
        "operation": "fit",
        "device": "cpu",
        "modelLabel": "returns.daily",
        "modelConfig": {"seqLen": 2, "hidden": 8, "nhead": 2},
        "trainingConfig": {"epochs": 1},
    }
    document.update(overrides)
    return document


@pytest.fixture
def coordinator(tmp_path):
    config = FlightServiceConfig(
        state_dir=str(tmp_path),
        port=0,
        profile="development",
        allow_plaintext=True,
        disk_min_free_bytes=1,
    ).validate()
    spool = Spool(config.state_dir).initialize()
    ledger = Ledger(config.database_path).initialize()
    return JobCoordinator(
        config,
        ledger,
        spool,
        cuda_available=lambda: False,
    ), ledger


def test_create_and_status_are_durable_and_idempotent(coordinator):
    service, ledger = coordinator
    document = create_document()
    request = validate_action_request(CREATE_ACTION, document)

    first = service.create("inventory", request, document)
    service.set_draining(True)
    retry_document = {**document, "requestId": str(uuid.uuid4())}
    retry = service.create(
        "inventory",
        validate_action_request(CREATE_ACTION, retry_document),
        retry_document,
    )

    assert retry == first
    assert len(ledger.list_jobs()) == 1
    status = service.status("inventory", first["jobId"], str(uuid.uuid4()))
    assert status["state"] == "UPLOADING"
    assert status["revision"] == 1
    assert status["committedInputs"] == []


def test_status_hides_legacy_cancel_error_after_ledger_reopen(tmp_path):
    config = FlightServiceConfig(
        state_dir=str(tmp_path),
        port=0,
        profile="development",
        allow_plaintext=True,
        disk_min_free_bytes=1,
    ).validate()
    spool = Spool(config.state_dir).initialize()
    ledger = Ledger(config.database_path).initialize()
    job_id = "b257793b-100f-4322-b28a-8c47072b7fec"
    ledger.create_job(
        job_id=job_id,
        owner_subject="inventory",
        operation="fit",
        requested_device="cuda",
        prediction_column="out",
        config_hash="a" * 64,
        model_label="legacy-cancelled",
        now=1.0,
    )
    ledger.seal_job(
        job_id,
        manifest_hash="b" * 64,
        manifest=[],
        now=2.0,
    )
    ledger.queue_job(job_id, selected_device="cuda", now=3.0)
    running = ledger.claim_next_job("cuda", now=4.0)
    ledger.transition_job(job_id, JobState.CANCELLING, now=5.0)
    ledger.finish_attempt(
        job_id,
        running["attempt"],
        JobState.CANCELLED,
        now=6.0,
    )
    with ledger.connection() as connection:
        connection.execute(
            "UPDATE jobs SET error_code='CANCELLED', error_message=? WHERE job_id=?",
            ("job was cancelled", job_id),
        )
        connection.execute(
            "UPDATE job_attempts SET error_code='CANCELLED', error_message=? "
            "WHERE job_id=? AND attempt=1",
            ("job was cancelled", job_id),
        )
        connection.commit()

    reopened = Ledger(config.database_path).initialize()
    service = JobCoordinator(
        config,
        reopened,
        spool,
        cuda_available=lambda: False,
    )
    status = service.status("inventory", job_id, str(uuid.uuid4()))

    assert status["state"] == JobState.CANCELLED.value
    assert status["error"] is None
    assert status["pollAfterMs"] == 0
    # Compatibility is a wire concern: immutable terminal storage is not
    # rewritten and its revision/timestamps remain untouched on startup.
    assert reopened.get_job(job_id)["error_code"] == "CANCELLED"


def test_same_idempotency_key_with_different_request_conflicts(coordinator):
    service, _ = coordinator
    document = create_document()
    service.create(
        "inventory",
        validate_action_request(CREATE_ACTION, document),
        document,
    )
    conflict = {**document, "modelLabel": "other"}

    with pytest.raises(ServiceError, match="different request"):
        service.create(
            "inventory",
            validate_action_request(CREATE_ACTION, conflict),
            conflict,
        )


def test_explicit_cuda_is_rejected_before_dataset_upload_without_cpu_fallback(coordinator):
    service, ledger = coordinator
    document = create_document(device="cuda")

    with pytest.raises(ServiceError) as error:
        service.create(
            "inventory",
            validate_action_request(CREATE_ACTION, document),
            document,
        )

    assert error.value.code.value == "DEVICE_UNAVAILABLE"
    assert ledger.list_jobs() == []


def test_cuda_unavailable_does_not_make_liveness_false(coordinator):
    service, _ = coordinator
    result = service.health(str(uuid.uuid4()))
    capabilities = service.capabilities(str(uuid.uuid4()))

    assert result["live"] is True
    assert result["cuda"] == {"available": False}
    assert capabilities["devices"]["cuda"]["available"] is False
    assert capabilities["features"] == {
        "doExchange": False,
        "pollFlightInfo": False,
    }


def test_failed_ledger_probe_changes_readiness_not_liveness(coordinator, monkeypatch):
    service, ledger = coordinator
    monkeypatch.setattr(
        ledger,
        "healthcheck",
        lambda: (_ for _ in ()).throw(OSError("database unavailable")),
    )

    result = service.health(str(uuid.uuid4()))

    assert result["live"] is True
    assert result["ready"] is False
    assert result["ledger"] == {"available": False}


def test_wrong_owner_cannot_observe_job(coordinator):
    service, _ = coordinator
    document = create_document()
    created = service.create(
        "inventory-a",
        validate_action_request(CREATE_ACTION, document),
        document,
    )

    with pytest.raises(ServiceError, match="job not found"):
        service.status("inventory-b", created["jobId"], str(uuid.uuid4()))


def commit_input(ledger, job_id, ordinal, payload_id, digest):
    token = f"upload-{ordinal}"
    ledger.reserve_input(
        job_id=job_id,
        payload_id=payload_id,
        ordinal=ordinal,
        upload_token=token,
        temporary_path=f"spool/jobs/{job_id}/inputs/.{ordinal}.tmp",
    )
    ledger.commit_input(
        upload_token=token,
        relative_path=f"spool/jobs/{job_id}/inputs/{ordinal}.arrow",
        schema_id="inventory.sequence.fit.v1",
        rows=1,
        batches=1,
        byte_count=100,
        sha256=digest,
        schema_fingerprint="f" * 64,
        source_width=4,
        feature_dim=2,
        max_payloads=10,
        max_job_bytes=10_000,
    )


def test_seal_uses_ordinal_manifest_and_replays_lost_response(coordinator):
    service, ledger = coordinator
    create = create_document()
    created = service.create(
        "inventory",
        validate_action_request(CREATE_ACTION, create),
        create,
    )
    payload_zero = str(uuid.uuid4())
    payload_one = str(uuid.uuid4())
    # RPC completion order is intentionally the reverse of semantic ordinal.
    commit_input(ledger, created["jobId"], 1, payload_one, "b" * 64)
    commit_input(ledger, created["jobId"], 0, payload_zero, "a" * 64)
    seal = {
        **common(),
        "idempotencyKey": "seal-1",
        "jobId": created["jobId"],
        "manifest": [
            {"payloadId": payload_zero, "ordinal": 0, "sha256": "a" * 64},
            {"payloadId": payload_one, "ordinal": 1, "sha256": "b" * 64},
        ],
    }

    first = service.seal(
        "inventory",
        validate_action_request(SEAL_ACTION, seal),
        seal,
    )
    retry = {**seal, "requestId": str(uuid.uuid4())}
    repeated = service.seal(
        "inventory",
        validate_action_request(SEAL_ACTION, retry),
        retry,
    )

    assert repeated == first
    assert first["state"] == "SEALED"
    assert [row["ordinal"] for row in ledger.list_inputs(created["jobId"])] == [0, 1]


def test_seal_rejects_missing_or_out_of_order_ordinal(coordinator):
    service, ledger = coordinator
    create = create_document(idempotencyKey="create-missing")
    created = service.create(
        "inventory",
        validate_action_request(CREATE_ACTION, create),
        create,
    )
    payload = str(uuid.uuid4())
    commit_input(ledger, created["jobId"], 1, payload, "b" * 64)
    seal = {
        **common(),
        "idempotencyKey": "seal-missing",
        "jobId": created["jobId"],
        "manifest": [{"payloadId": payload, "ordinal": 1, "sha256": "b" * 64}],
    }

    with pytest.raises(ServiceError, match="contiguous from zero"):
        service.seal(
            "inventory",
            validate_action_request(SEAL_ACTION, seal),
            seal,
        )
    assert ledger.get_job(created["jobId"])["state"] == "UPLOADING"


def test_start_rechecks_explicit_cuda_and_preserves_sealed_state(coordinator):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cuda",
        prediction_column="out",
        config_hash="a" * 64,
        model_label="cuda-model",
        model_config={"seq_len": 2, "feature_dim": 2},
    )
    ledger.seal_job(job["job_id"], manifest_hash="a" * 64, manifest=[])
    start = {
        **common(),
        "idempotencyKey": "start-cuda",
        "jobId": job["job_id"],
    }

    with pytest.raises(ServiceError) as error:
        service.start(
            "inventory",
            validate_action_request(START_ACTION, start),
            start,
        )
    assert error.value.code.value == "DEVICE_UNAVAILABLE"
    assert ledger.get_job(job["job_id"])["state"] == "SEALED"


def test_start_replays_lost_response_and_rejects_conflicting_repeat(coordinator):
    service, ledger = coordinator
    first = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        model_label="first-model",
    )
    ledger.seal_job(first["job_id"], manifest_hash="a" * 64, manifest=[])
    start = {
        **common(),
        "idempotencyKey": "start-lost-response",
        "jobId": first["job_id"],
    }

    result = service.start(
        "inventory",
        validate_action_request(START_ACTION, start),
        start,
    )
    retry = {**start, "requestId": str(uuid.uuid4())}
    repeated = service.start(
        "inventory",
        validate_action_request(START_ACTION, retry),
        retry,
    )

    assert repeated == result
    assert result["state"] == "QUEUED"
    assert ledger.get_job(first["job_id"])["attempt"] == 0

    second = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="b" * 64,
        model_label="second-model",
    )
    ledger.seal_job(second["job_id"], manifest_hash="b" * 64, manifest=[])
    conflicting = {**retry, "jobId": second["job_id"]}
    with pytest.raises(ServiceError, match="different request"):
        service.start(
            "inventory",
            validate_action_request(START_ACTION, conflicting),
            conflicting,
        )
    assert ledger.get_job(second["job_id"])["state"] == "SEALED"


def test_lifecycle_repeats_with_new_keys_return_original_result_after_progress(
    coordinator,
):
    service, ledger = coordinator
    create = create_document(idempotencyKey="create-lifecycle-repeat")
    created = service.create(
        "inventory",
        validate_action_request(CREATE_ACTION, create),
        create,
    )
    seal = {
        **common(),
        "idempotencyKey": "seal-lifecycle-first",
        "jobId": created["jobId"],
        "manifest": [],
    }
    sealed = service.seal(
        "inventory",
        validate_action_request(SEAL_ACTION, seal),
        seal,
    )
    start = {
        **common(),
        "idempotencyKey": "start-lifecycle-first",
        "jobId": created["jobId"],
    }
    queued = service.start(
        "inventory",
        validate_action_request(START_ACTION, start),
        start,
    )
    assert ledger.claim_next_job("cpu")["state"] == "RUNNING"

    repeated_seal_document = {
        **seal,
        "requestId": str(uuid.uuid4()),
        "idempotencyKey": "seal-lifecycle-second",
    }
    repeated_seal = service.seal(
        "inventory",
        validate_action_request(SEAL_ACTION, repeated_seal_document),
        repeated_seal_document,
    )
    repeated_start_document = {
        **start,
        "requestId": str(uuid.uuid4()),
        "idempotencyKey": "start-lifecycle-second",
    }
    repeated_start = service.start(
        "inventory",
        validate_action_request(START_ACTION, repeated_start_document),
        repeated_start_document,
    )

    assert repeated_seal == {
        **sealed,
        "requestId": repeated_seal_document["requestId"],
    }
    assert repeated_start == {
        **queued,
        "requestId": repeated_start_document["requestId"],
    }
    assert repeated_seal["state"] == "SEALED"
    assert repeated_start["state"] == "QUEUED"
    assert ledger.get_job(created["jobId"])["state"] == "RUNNING"


def test_draining_rejects_new_start_but_preserves_recorded_replay(coordinator):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        model_label="draining-start",
    )
    ledger.seal_job(job["job_id"], manifest_hash="a" * 64, manifest=[])
    start = {
        **common(),
        "idempotencyKey": "start-before-draining",
        "jobId": job["job_id"],
    }

    service.set_draining(True)
    with pytest.raises(ServiceError) as error:
        service.start(
            "inventory",
            validate_action_request(START_ACTION, start),
            start,
        )
    assert error.value.code.value == "UNAVAILABLE"
    assert ledger.get_job(job["job_id"])["state"] == "SEALED"

    service.set_draining(False)
    first = service.start(
        "inventory",
        validate_action_request(START_ACTION, start),
        start,
    )
    service.set_draining(True)
    replay = service.start(
        "inventory",
        validate_action_request(START_ACTION, start),
        start,
    )
    assert replay == first


@pytest.mark.parametrize("initial_state", ["UPLOADING", "SEALED", "QUEUED"])
def test_cancel_is_immediate_idempotent_and_terminal_state_is_immutable(
    coordinator,
    initial_state,
):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="c" * 64,
        model_label=f"cancel-{initial_state.lower()}",
    )
    if initial_state in ("SEALED", "QUEUED"):
        ledger.seal_job(job["job_id"], manifest_hash="c" * 64, manifest=[])
    if initial_state == "QUEUED":
        ledger.queue_job(job["job_id"], selected_device="cpu")
    revision = ledger.get_job(job["job_id"])["revision"]
    cancel = {
        **common(),
        "idempotencyKey": f"cancel-{initial_state.lower()}-lost-response",
        "jobId": job["job_id"],
    }

    result = service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, cancel),
        cancel,
    )
    retry = {**cancel, "requestId": str(uuid.uuid4())}
    repeated = service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, retry),
        retry,
    )

    assert repeated == result
    assert result["state"] == "CANCELLED"
    assert result["revision"] == revision + 1
    terminal = ledger.get_job(job["job_id"])
    assert terminal["state"] == "CANCELLED"
    terminal_revision = terminal["revision"]

    another_cancel = {
        **common(),
        "idempotencyKey": f"cancel-{initial_state.lower()}-terminal",
        "jobId": job["job_id"],
    }
    terminal_result = service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, another_cancel),
        another_cancel,
    )
    assert terminal_result["state"] == "CANCELLED"
    assert terminal_result["revision"] == terminal_revision
    assert ledger.get_job(job["job_id"])["revision"] == terminal_revision


def test_cancel_rejects_same_idempotency_key_for_different_job(coordinator):
    service, ledger = coordinator

    def create(label):
        return ledger.create_job(
            job_id=str(uuid.uuid4()),
            owner_subject="inventory",
            operation="fit",
            requested_device="cpu",
            prediction_column="out",
            config_hash="d" * 64,
            model_label=label,
        )

    first = create("cancel-first")
    second = create("cancel-second")
    cancel = {
        **common(),
        "idempotencyKey": "cancel-conflicting-job",
        "jobId": first["job_id"],
    }
    service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, cancel),
        cancel,
    )
    conflicting = {
        **cancel,
        "requestId": str(uuid.uuid4()),
        "jobId": second["job_id"],
    }

    with pytest.raises(ServiceError, match="different request"):
        service.cancel(
            "inventory",
            validate_action_request(CANCEL_ACTION, conflicting),
            conflicting,
        )
    assert ledger.get_job(second["job_id"])["state"] == "UPLOADING"


def test_cancel_commit_before_result_publication_prevents_success(coordinator):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="e" * 64,
        input_model_ref="mdl_test",
    )
    ledger.seal_job(job["job_id"], manifest_hash="e" * 64, manifest=[])
    ledger.queue_job(job["job_id"], selected_device="cpu")
    running = ledger.claim_next_job("cpu")
    cancel = {
        **common(),
        "idempotencyKey": "cancel-before-publication",
        "jobId": job["job_id"],
    }

    response = service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, cancel),
        cancel,
    )

    assert response["state"] == "CANCELLING"
    with pytest.raises(ServiceError, match="active running attempt"):
        ledger.publish_outputs(
            job["job_id"],
            running["attempt"],
            [{
                "ordinal": 0,
                "rows": 0,
                "batches": 0,
                "bytes": 1,
                "sha256": "a" * 64,
                "schema_fingerprint": "b" * 64,
                "relative_path": (
                    f"spool/jobs/{job['job_id']}/attempts/1/outputs/0.arrow"
                ),
            }],
            result={"outputs": [{"ordinal": 0}]},
        )
    assert ledger.list_outputs(job["job_id"]) == []


def test_result_publication_commit_before_cancel_remains_successful(coordinator):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="f" * 64,
        input_model_ref="mdl_test",
    )
    ledger.seal_job(job["job_id"], manifest_hash="f" * 64, manifest=[])
    ledger.queue_job(job["job_id"], selected_device="cpu")
    running = ledger.claim_next_job("cpu")
    succeeded = ledger.publish_outputs(
        job["job_id"],
        running["attempt"],
        [{
            "ordinal": 0,
            "rows": 0,
            "batches": 0,
            "bytes": 1,
            "sha256": "a" * 64,
            "schema_fingerprint": "b" * 64,
            "relative_path": (
                f"spool/jobs/{job['job_id']}/attempts/1/outputs/0.arrow"
            ),
        }],
        result={"outputs": [{"ordinal": 0}]},
    )
    cancel = {
        **common(),
        "idempotencyKey": "cancel-after-publication",
        "jobId": job["job_id"],
    }

    response = service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, cancel),
        cancel,
    )

    assert response["state"] == "SUCCEEDED"
    assert response["revision"] == succeeded["revision"]
    assert ledger.get_job(job["job_id"])["state"] == "SUCCEEDED"
    assert len(ledger.list_outputs(job["job_id"])) == 1
