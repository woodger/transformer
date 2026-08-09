import json
import threading
import time
import uuid

import pyarrow.flight as flight
import pytest
from flight_contract_schema import (
    read_contract_schema,
    validate_contract_document,
)
from sqlalchemy import event as sqlalchemy_event, text

from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CANCEL_ACTION,
    CONTRACT_NAME,
    CREATE_ACTION,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
    ErrorCode,
    JobState,
)
from app.flight.contract import validate_action_request
from app.flight.coordinator import JobCoordinator
from app.flight.errors import ServiceError
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool

ACTION_RESULT_SCHEMAS = {
    CREATE_ACTION: "create-result.schema.json",
    SEAL_ACTION: "seal-result.schema.json",
    START_ACTION: "start-result.schema.json",
    CANCEL_ACTION: "cancel-result.schema.json",
    STATUS_ACTION: "status-result.schema.json",
}


def _is_job_select(statement: str) -> bool:
    normalized = " ".join(statement.lower().split())
    return (
        normalized.startswith("select")
        and "owner_subject" in normalized
        and "jobs" in normalized
    )


def _wait_for_postgres_lock(
    database,
    backend_pid: int,
    *,
    operation_done,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if operation_done():
            pytest.fail(
                "concurrent operation completed before waiting for "
                "the PostgreSQL row lock"
            )
        with database.engine.connect() as connection:
            wait_event_type = connection.scalar(
                text(
                    "SELECT wait_event_type "
                    "FROM pg_stat_activity WHERE pid = :pid"
                ),
                {"pid": backend_pid},
            )
        if wait_event_type == "Lock":
            return
        time.sleep(0.01)
    pytest.fail("concurrent PostgreSQL operation did not wait for a row lock")


def _thread_call(target, *, name):
    results = []
    errors = []
    finished = threading.Event()

    def invoke():
        try:
            results.append(target())
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    return (
        threading.Thread(target=invoke, name=name),
        results,
        errors,
        finished,
    )


def _join_thread(thread, *, timeout=5.0):
    if thread.ident is not None:
        thread.join(timeout=timeout)


def _running_predict_job(ledger, digest):
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash=digest,
        input_model_ref="mdl_test",
    )
    ledger.seal_job(
        job["job_id"],
        manifest_hash=digest,
        manifest=[],
    )
    ledger.queue_job(job["job_id"], selected_device="cpu")
    return job, ledger.claim_next_job("cpu")


def _prediction_output(job_id, *, rows=0):
    return {
        "ordinal": 0,
        "rows": rows,
        "batches": int(rows > 0),
        "bytes": 100,
        "sha256": "a" * 64,
        "schema_fingerprint": "b" * 64,
        "relative_path": (
            f"spool/jobs/{job_id}/attempts/1/outputs/0.arrow"
        ),
    }


def common():
    return {
        "contract": CONTRACT_NAME,
        "version": 2,
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
def coordinator(tmp_path, postgres_ledger):
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        port=0,
        allow_plaintext=True,
    ).validate()
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    ledger = postgres_ledger
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


def test_action_facade_preserves_lifecycle_over_real_flight_loopback(
    coordinator,
):
    service, _ = coordinator
    server = TransformerFlightServer(
        service.config,
        service,
        {"secret": "inventory"},
    )
    client = flight.FlightClient(("localhost", server.port))
    options = flight.FlightCallOptions(
        headers=[(b"authorization", b"Bearer secret")],
        timeout=5.0,
    )

    def action(action_name, document):
        results = list(client.do_action(
            flight.Action(
                action_name,
                json.dumps(document).encode("utf-8"),
            ),
            options=options,
        ))
        assert len(results) == 1
        response = json.loads(results[0].body.to_pybytes())
        validate_contract_document(
            response,
            read_contract_schema(ACTION_RESULT_SCHEMAS[action_name]),
        )
        return response

    try:
        create = {
            **common(),
            "idempotencyKey": "loopback-create",
            "operation": "fit",
            "device": "cpu",
            "modelLabel": "loopback",
            "modelConfig": {
                "seqLen": 2,
                "hidden": 8,
                "nhead": 2,
            },
            "trainingConfig": {"epochs": 1},
        }
        created = action(CREATE_ACTION, create)
        assert created["requestId"] == create["requestId"]
        assert action(
            CREATE_ACTION,
            {**create, "requestId": str(uuid.uuid4())},
        ) == created
        job_id = created["jobId"]
        uploading_status = {**common(), "jobId": job_id}
        uploading = action(STATUS_ACTION, uploading_status)
        assert uploading["requestId"] == uploading_status["requestId"]
        assert uploading["state"] == "UPLOADING"

        seal = {
            **common(),
            "idempotencyKey": "loopback-seal",
            "jobId": job_id,
            "manifest": [],
        }
        sealed = action(SEAL_ACTION, seal)
        assert sealed["requestId"] == seal["requestId"]
        assert sealed["state"] == "SEALED"
        assert action(
            SEAL_ACTION,
            {**seal, "requestId": str(uuid.uuid4())},
        ) == sealed

        start = {
            **common(),
            "idempotencyKey": "loopback-start",
            "jobId": job_id,
        }
        started = action(START_ACTION, start)
        assert started["requestId"] == start["requestId"]
        assert started["state"] == "QUEUED"
        assert action(
            START_ACTION,
            {**start, "requestId": str(uuid.uuid4())},
        ) == started
        queued_status = {**common(), "jobId": job_id}
        queued = action(STATUS_ACTION, queued_status)
        assert queued["requestId"] == queued_status["requestId"]
        assert queued["state"] == "QUEUED"

        cancel = {
            **common(),
            "idempotencyKey": "loopback-cancel",
            "jobId": job_id,
        }
        cancelled = action(CANCEL_ACTION, cancel)
        assert cancelled["requestId"] == cancel["requestId"]
        assert cancelled["state"] == "CANCELLED"
        assert action(
            CANCEL_ACTION,
            {**cancel, "requestId": str(uuid.uuid4())},
        ) == cancelled
        terminal_status = {**common(), "jobId": job_id}
        terminal = action(STATUS_ACTION, terminal_status)
        assert terminal["requestId"] == terminal_status["requestId"]
        assert terminal["state"] == "CANCELLED"
        assert terminal["error"] is None
    finally:
        client.close()
        server.shutdown()


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
    assert result["cuda"] == {
        "available": False,
        "deviceCount": 0,
        "quarantinedCount": 0,
    }
    assert capabilities["devices"]["cuda"]["available"] is False
    assert capabilities["queue"]["cudaCapacity"] == 0
    assert capabilities["features"] == {
        "doExchange": False,
        "pollFlightInfo": False,
        "resumableFit": True,
        "recoveryBoundary": "globalEpoch",
        "deviceAwareCuda": True,
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


def test_storage_free_bytes_are_telemetry_not_readiness_policy(
    coordinator,
    monkeypatch,
):
    service, _ = coordinator
    monkeypatch.setattr(
        service.spool,
        "disk_usage",
        lambda: type("Usage", (), {
            "total": 100,
            "used": 100,
            "free": 0,
        })(),
    )

    result = service.health(str(uuid.uuid4()))

    assert result["ready"] is True
    assert result["storage"] == {
        "runtime": {"freeBytes": 0},
        "recovery": {"freeBytes": 0},
    }


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


def test_seal_serializes_manifest_validation_with_input_commit(
    coordinator,
    monkeypatch,
):
    service, ledger = coordinator
    create = create_document(idempotencyKey="create-seal-race")
    created = service.create(
        "inventory",
        validate_action_request(CREATE_ACTION, create),
        create,
    )
    first_payload = str(uuid.uuid4())
    second_payload = str(uuid.uuid4())
    commit_input(
        ledger,
        created["jobId"],
        0,
        first_payload,
        "a" * 64,
    )
    ledger.reserve_input(
        job_id=created["jobId"],
        payload_id=second_payload,
        ordinal=1,
        upload_token="upload-seal-race",
        temporary_path=(
            f"spool/jobs/{created['jobId']}/inputs/.1.tmp"
        ),
    )
    seal = {
        **common(),
        "idempotencyKey": "seal-race",
        "jobId": created["jobId"],
        "manifest": [{
            "payloadId": first_payload,
            "ordinal": 0,
            "sha256": "a" * 64,
        }],
    }

    snapshot_read = threading.Event()
    continue_seal = threading.Event()
    original_list_inputs = ledger.list_inputs

    def paused_list_inputs(*args, **kwargs):
        inputs = original_list_inputs(*args, **kwargs)
        if not snapshot_read.is_set():
            snapshot_read.set()
            if not continue_seal.wait(timeout=5):
                raise TimeoutError("seal race test did not resume")
        return inputs

    monkeypatch.setattr(ledger, "list_inputs", paused_list_inputs)
    seal_errors = []
    commit_errors = []

    def seal_job():
        try:
            service.seal(
                "inventory",
                validate_action_request(SEAL_ACTION, seal),
                seal,
            )
        except BaseException as exc:
            seal_errors.append(exc)

    def commit_second_input():
        try:
            ledger.commit_input(
                upload_token="upload-seal-race",
                relative_path=(
                    f"spool/jobs/{created['jobId']}/inputs/1.arrow"
                ),
                schema_id="inventory.sequence.fit.v1",
                rows=1,
                batches=1,
                byte_count=100,
                sha256="b" * 64,
                schema_fingerprint="f" * 64,
                source_width=4,
                feature_dim=2,
                max_payloads=10,
                max_job_bytes=10_000,
            )
        except BaseException as exc:
            commit_errors.append(exc)

    seal_thread = threading.Thread(target=seal_job)
    commit_thread = threading.Thread(target=commit_second_input)
    seal_thread.start()
    try:
        assert snapshot_read.wait(timeout=5)
        commit_thread.start()
        commit_thread.join(timeout=0.5)
    finally:
        continue_seal.set()
        seal_thread.join(timeout=5)
        if commit_thread.ident is not None:
            commit_thread.join(timeout=5)

    assert not seal_thread.is_alive()
    assert not commit_thread.is_alive()
    assert commit_errors == []
    assert len(seal_errors) == 1
    assert isinstance(seal_errors[0], ServiceError)
    assert "upload in progress" in str(seal_errors[0])
    assert ledger.get_job(created["jobId"])["state"] == "UPLOADING"
    assert [
        item["ordinal"]
        for item in original_list_inputs(created["jobId"])
    ] == [0, 1]


@pytest.mark.parametrize(
    (
        "operation",
        "action_name",
        "ledger_method",
        "initial_state",
        "expected_state",
    ),
    [
        (
            "seal",
            SEAL_ACTION,
            "seal_job",
            "UPLOADING",
            "SEALED",
        ),
        (
            "start",
            START_ACTION,
            "queue_job",
            "SEALED",
            "QUEUED",
        ),
    ],
)
def test_lifecycle_slice_and_idempotency_record_roll_back_together(
    coordinator,
    monkeypatch,
    operation,
    action_name,
    ledger_method,
    initial_state,
    expected_state,
):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        model_label=f"{operation}-rollback",
        model_config={"seq_len": 2},
    )
    if operation == "start":
        ledger.seal_job(
            job["job_id"],
            manifest_hash="b" * 64,
            manifest=[],
        )
    document = {
        **common(),
        "idempotencyKey": f"{operation}-rollback",
        "jobId": job["job_id"],
    }
    if operation == "seal":
        document["manifest"] = []
    request = validate_action_request(action_name, document)
    original = getattr(ledger, ledger_method)

    def mutate_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-mutation failure")

    monkeypatch.setattr(ledger, ledger_method, mutate_then_fail)
    with pytest.raises(
        RuntimeError,
        match="injected post-mutation failure",
    ):
        getattr(service, operation)("inventory", request, document)

    assert ledger.get_job(job["job_id"])["state"] == initial_state
    assert ledger.lookup_idempotency(
        "inventory",
        action_name,
        document["idempotencyKey"],
    ) is None

    monkeypatch.setattr(ledger, ledger_method, original)
    result = getattr(service, operation)("inventory", request, document)

    assert result["state"] == expected_state
    assert ledger.get_job(job["job_id"])["state"] == expected_state


def test_create_slice_and_idempotency_record_roll_back_together(
    coordinator,
    monkeypatch,
):
    service, ledger = coordinator
    document = create_document(
        idempotencyKey="create-rollback",
        modelLabel="create-rollback",
    )
    request = validate_action_request(CREATE_ACTION, document)
    original = ledger.create_job

    def mutate_then_fail(**kwargs):
        original(**kwargs)
        raise RuntimeError("injected post-mutation failure")

    monkeypatch.setattr(ledger, "create_job", mutate_then_fail)
    with pytest.raises(
        RuntimeError,
        match="injected post-mutation failure",
    ):
        service.create("inventory", request, document)

    assert ledger.list_jobs() == []
    assert ledger.lookup_idempotency(
        "inventory",
        CREATE_ACTION,
        document["idempotencyKey"],
    ) is None

    monkeypatch.setattr(ledger, "create_job", original)
    result = service.create("inventory", request, document)

    assert result["state"] == "UPLOADING"
    assert ledger.get_job(result["jobId"])["state"] == "UPLOADING"


def test_running_cancel_rolls_back_with_idempotency_and_notifier(
    coordinator,
    monkeypatch,
):
    service, ledger = coordinator
    job = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="d" * 64,
        model_label="cancel-rollback",
    )
    ledger.seal_job(
        job["job_id"],
        manifest_hash="e" * 64,
        manifest=[],
    )
    ledger.queue_job(job["job_id"], selected_device="cpu")
    assert ledger.claim_next_job("cpu")["state"] == "RUNNING"
    notifications = []
    service.cancel_notifier = notifications.append
    document = {
        **common(),
        "idempotencyKey": "cancel-running-rollback",
        "jobId": job["job_id"],
    }
    request = validate_action_request(CANCEL_ACTION, document)
    original = ledger.transition_job

    def mutate_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-mutation failure")

    monkeypatch.setattr(ledger, "transition_job", mutate_then_fail)
    with pytest.raises(
        RuntimeError,
        match="injected post-mutation failure",
    ):
        service.cancel("inventory", request, document)

    assert ledger.get_job(job["job_id"])["state"] == "RUNNING"
    assert ledger.lookup_idempotency(
        "inventory",
        CANCEL_ACTION,
        document["idempotencyKey"],
    ) is None
    assert notifications == []

    monkeypatch.setattr(ledger, "transition_job", original)
    result = service.cancel("inventory", request, document)

    assert result["state"] == "CANCELLING"
    assert ledger.get_job(job["job_id"])["state"] == "CANCELLING"
    assert notifications == [job["job_id"]]
    retry = {
        **document,
        "requestId": str(uuid.uuid4()),
    }
    assert service.cancel(
        "inventory",
        validate_action_request(CANCEL_ACTION, retry),
        retry,
    ) == result
    assert notifications == [job["job_id"]]


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
    job, running = _running_predict_job(ledger, "e" * 64)
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
            [_prediction_output(job["job_id"])],
            attempt_id=running["attempt_id"],
            result={"outputs": [{"ordinal": 0}]},
        )
    assert ledger.list_outputs(job["job_id"]) == []


def test_result_publication_commit_before_cancel_remains_successful(coordinator):
    service, ledger = coordinator
    job, running = _running_predict_job(ledger, "f" * 64)
    succeeded = ledger.publish_outputs(
        job["job_id"],
        running["attempt"],
        [_prediction_output(job["job_id"])],
        attempt_id=running["attempt_id"],
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


def test_cancel_and_result_publication_have_one_atomic_winner(
    coordinator,
):
    service, ledger = coordinator
    job, running = _running_predict_job(ledger, "1" * 64)
    output = _prediction_output(job["job_id"], rows=1)
    cancel = {
        **common(),
        "idempotencyKey": "cancel-publication-race",
        "jobId": job["job_id"],
    }
    backend_pids = set()
    both_locks_attempted = threading.Event()
    pid_lock = threading.Lock()

    def capture_job_lock_attempt(
        connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if (
            not threading.current_thread().name.startswith(
                "cancel-publication-race"
            )
            or not _is_job_select(statement)
            or "FOR UPDATE" not in statement.upper()
        ):
            return
        with pid_lock:
            backend_pids.add(
                connection.connection.driver_connection.info.backend_pid
            )
            if len(backend_pids) == 2:
                both_locks_attempted.set()

    sqlalchemy_event.listen(
        ledger.database.engine,
        "before_cursor_execute",
        capture_job_lock_attempt,
    )

    cancel_thread, cancel_results, cancel_errors, cancel_finished = (
        _thread_call(
            lambda: service.cancel(
                "inventory",
                validate_action_request(CANCEL_ACTION, cancel),
                cancel,
            ),
            name="cancel-publication-race-cancel",
        )
    )
    (
        publication_thread,
        publication_results,
        publication_errors,
        publication_finished,
    ) = _thread_call(
        lambda: ledger.publish_outputs(
            job["job_id"],
            running["attempt"],
            [output],
            attempt_id=running["attempt_id"],
            result={"outputs": [{"ordinal": 0}]},
        ),
        name="cancel-publication-race-publish",
    )
    try:
        with ledger.transaction() as blocker:
            assert ledger.get_job(
                job["job_id"],
                connection=blocker,
                for_update=True,
            )["state"] == JobState.RUNNING.value
            cancel_thread.start()
            publication_thread.start()
            assert both_locks_attempted.wait(timeout=5)
            for backend_pid in backend_pids:
                _wait_for_postgres_lock(
                    ledger.database,
                    backend_pid,
                    operation_done=lambda: (
                        cancel_finished.is_set()
                        or publication_finished.is_set()
                    ),
                )
    finally:
        _join_thread(cancel_thread)
        _join_thread(publication_thread)
        sqlalchemy_event.remove(
            ledger.database.engine,
            "before_cursor_execute",
            capture_job_lock_attempt,
        )

    assert not cancel_thread.is_alive()
    assert not publication_thread.is_alive()
    assert cancel_errors == []
    assert len(cancel_results) == 1
    cancel_result = cancel_results[0]
    final_job = ledger.get_job(job["job_id"])
    published_outputs = ledger.list_outputs(job["job_id"])
    if publication_results:
        assert publication_errors == []
        assert cancel_result["state"] == JobState.SUCCEEDED.value
        assert final_job["state"] == JobState.SUCCEEDED.value
        assert [item["ordinal"] for item in published_outputs] == [0]
    else:
        assert len(publication_errors) == 1
        assert isinstance(publication_errors[0], ServiceError)
        assert publication_errors[0].code == ErrorCode.FAILED_PRECONDITION
        assert cancel_result["state"] == JobState.CANCELLING.value
        assert final_job["state"] == JobState.CANCELLING.value
        assert published_outputs == []


def test_status_snapshot_is_consistent_while_result_is_published(
    coordinator,
):
    service, ledger = coordinator
    job, running = _running_predict_job(ledger, "4" * 64)
    output = _prediction_output(job["job_id"], rows=1)
    snapshot_job_read = threading.Event()
    release_snapshot = threading.Event()
    publication_lock_attempted = threading.Event()
    publication_pid = []

    def pause_status_after_job_read(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if (
            threading.current_thread().name
            != "status-publication-status"
            or not _is_job_select(statement)
            or snapshot_job_read.is_set()
        ):
            return
        snapshot_job_read.set()
        if not release_snapshot.wait(timeout=5):
            raise TimeoutError("status snapshot query was not released")

    def capture_publication_lock_attempt(
        connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if (
            threading.current_thread().name
            == "status-publication-publish"
            and _is_job_select(statement)
            and "FOR UPDATE" in statement.upper()
        ):
            publication_pid.append(
                connection.connection.driver_connection.info.backend_pid
            )
            publication_lock_attempted.set()

    (
        status_thread,
        snapshots,
        status_errors,
        _status_finished,
    ) = _thread_call(
        lambda: service.status(
            "inventory",
            job["job_id"],
            str(uuid.uuid4()),
        ),
        name="status-publication-status",
    )
    (
        publication_thread,
        publication_results,
        publication_errors,
        publication_finished,
    ) = _thread_call(
        lambda: ledger.publish_outputs(
            job["job_id"],
            running["attempt"],
            [output],
            attempt_id=running["attempt_id"],
            result={"outputs": [{"ordinal": 0}]},
        ),
        name="status-publication-publish",
    )
    sqlalchemy_event.listen(
        ledger.database.engine,
        "after_cursor_execute",
        pause_status_after_job_read,
    )
    sqlalchemy_event.listen(
        ledger.database.engine,
        "before_cursor_execute",
        capture_publication_lock_attempt,
    )
    try:
        status_thread.start()
        assert snapshot_job_read.wait(timeout=5)
        publication_thread.start()
        assert publication_lock_attempted.wait(timeout=5)
        _wait_for_postgres_lock(
            ledger.database,
            publication_pid[0],
            operation_done=publication_finished.is_set,
        )
    finally:
        release_snapshot.set()
        _join_thread(status_thread)
        _join_thread(publication_thread)
        sqlalchemy_event.remove(
            ledger.database.engine,
            "after_cursor_execute",
            pause_status_after_job_read,
        )
        sqlalchemy_event.remove(
            ledger.database.engine,
            "before_cursor_execute",
            capture_publication_lock_attempt,
        )

    assert not status_thread.is_alive()
    assert not publication_thread.is_alive()
    assert status_errors == []
    assert publication_errors == []
    assert len(publication_results) == 1
    assert len(snapshots) == 1
    assert snapshots[0]["state"] == JobState.RUNNING.value
    assert snapshots[0]["results"]["outputs"] == []
    assert ledger.get_job(job["job_id"])["state"] == JobState.SUCCEEDED.value
    assert [
        item["ordinal"]
        for item in ledger.list_outputs(job["job_id"])
    ] == [0]
