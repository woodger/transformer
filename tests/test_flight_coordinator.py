from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.contracts.worker.v3.objective import objective_config
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CREATE_ACTION,
    INPUT_CLOSE_ACTION,
    MODEL_DESCRIBE_ACTION,
    STATUS_ACTION,
    ErrorCode,
)
from app.service.adapters.inbound.flight.contract import validate_action_request
from app.service.adapters.outbound.artifact_storage.spool import Spool
from app.service.bootstrap.config import FlightServiceConfig
from app.service.domain.errors import ServiceError
from app.service.domain.input_manifest import manifest_sha256
from tests.flight_v4_helpers import (
    DATA_CONTRACT_SHA256,
    OWNER,
    build_test_job_coordinator,
    close_input,
    commit_input,
    create_fit,
    internal_data_contract,
    model_config,
    public_data_contract,
    public_ml_contract,
    train_config,
)


@pytest.fixture
def coordinator_components(tmp_path, postgres_ledger):
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        allow_plaintext=True,
    ).validate()
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    coordinator = build_test_job_coordinator(
        config,
        postgres_ledger,
        spool,
        cuda_available=lambda: False,
    )
    return config, spool, postgres_ledger, coordinator


def _common():
    return {
        "contract": "transformer-flight",
        "version": 4,
        "requestId": str(uuid.uuid4()),
    }


def _fit_create(*, job_id=None, execution_id=None, idempotency_key=None):
    return {
        **_common(),
        "idempotencyKey": idempotency_key or f"fit-{uuid.uuid4()}",
        "jobId": job_id or str(uuid.uuid4()),
        "clientExecutionId": execution_id or str(uuid.uuid4()),
        "operation": "fit",
        "device": "cpu",
        "modelLabel": "daily",
        "modelConfig": {
            "seqLen": 2,
            "hidden": 8,
            "layers": 1,
            "dropout": 0.0,
            "nhead": 2,
            "mode": "relaxed",
        },
        "trainingConfig": {
            "batchSize": 2,
            "epochs": 2,
            "lossStage": 4,
            "lossSchedule": "none",
            "stageSize": 5,
            "directLossWeights": [1.0] * 6,
            "selection": None,
            "useAmp": False,
            "seed": 17,
            "deterministic": True,
        },
        "dataContract": public_data_contract(),
        "mlContract": public_ml_contract(),
    }


def _dispatch(coordinator, action, document, *, owner=OWNER):
    request = validate_action_request(action, document)
    return json.loads(
        coordinator.dispatch(action, owner, request, document)
    )


def test_capabilities_advertise_only_v4_streaming_surface(
    coordinator_components,
):
    _, _, _, coordinator = coordinator_components

    result = _dispatch(coordinator, CAPABILITIES_ACTION, _common())

    assert result["protocolVersions"] == [4]
    assert result["features"]["doExchange"] is False
    assert result["features"]["durableStreamingInput"] is True
    assert result["features"]["clientGeneratedJobId"] is True
    assert result["features"]["crossSystemFencing"] is True


def test_client_generated_identity_recovers_lost_create_response(
    coordinator_components,
):
    _, _, ledger, coordinator = coordinator_components
    job_id = str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    first = _fit_create(
        job_id=job_id,
        execution_id=execution_id,
        idempotency_key="first-network-attempt",
    )
    retry = {
        **first,
        "requestId": str(uuid.uuid4()),
        "idempotencyKey": "retry-after-lost-response",
    }

    first_result = _dispatch(coordinator, CREATE_ACTION, first)
    recovered = _dispatch(coordinator, CREATE_ACTION, retry)

    assert recovered == first_result
    assert recovered["jobId"] == job_id
    assert recovered["ownership"] == {
        "clientExecutionId": execution_id,
        "fencingToken": "1",
    }
    assert [job["job_id"] for job in ledger.list_jobs()] == [job_id]


def test_concurrent_create_with_same_identity_has_one_durable_result(
    coordinator_components,
):
    _, _, ledger, coordinator = coordinator_components
    job_id = str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    documents = [
        _fit_create(
            job_id=job_id,
            execution_id=execution_id,
            idempotency_key=f"concurrent-{index}",
        )
        for index in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda document: _dispatch(
                coordinator,
                CREATE_ACTION,
                document,
            ),
            documents,
        ))

    assert results[0] == results[1]
    assert [job["job_id"] for job in ledger.list_jobs()] == [job_id]


def test_reusing_job_identity_for_different_request_is_rejected(
    coordinator_components,
):
    _, _, _, coordinator = coordinator_components
    document = _fit_create()
    _dispatch(coordinator, CREATE_ACTION, document)
    changed = {
        **document,
        "requestId": str(uuid.uuid4()),
        "idempotencyKey": "changed-create",
        "modelLabel": "different",
    }

    with pytest.raises(ServiceError) as error:
        _dispatch(coordinator, CREATE_ACTION, changed)

    assert error.value.code is ErrorCode.ALREADY_EXISTS


def test_acquire_advances_decimal_fence_and_stale_mutations_fail(
    coordinator_components,
):
    _, _, ledger, coordinator = coordinator_components
    create = _fit_create()
    _dispatch(coordinator, CREATE_ACTION, create)
    new_execution_id = str(uuid.uuid4())
    acquire = {
        **_common(),
        "idempotencyKey": "takeover-1",
        "jobId": create["jobId"],
        "previousClientExecutionId": create["clientExecutionId"],
        "expectedFencingToken": "1",
        "clientExecutionId": new_execution_id,
    }

    result = _dispatch(coordinator, ACQUIRE_ACTION, acquire)

    assert result["ownership"] == {
        "clientExecutionId": new_execution_id,
        "fencingToken": "2",
    }
    stale_cancel = {
        **_common(),
        "idempotencyKey": "stale-cancel",
        "jobId": create["jobId"],
        "clientExecutionId": create["clientExecutionId"],
        "fencingToken": "1",
    }
    with pytest.raises(ServiceError) as error:
        _dispatch(coordinator, CANCEL_ACTION, stale_cancel)
    assert error.value.code is ErrorCode.STALE_FENCE
    assert ledger.get_job(create["jobId"])["fencing_token"] == 2


def test_input_close_is_eof_not_a_start_action(coordinator_components):
    _, _, ledger, coordinator = coordinator_components
    create = _fit_create()
    _dispatch(coordinator, CREATE_ACTION, create)
    job = ledger.get_job(create["jobId"])
    commit_input(ledger, job, 0, rows=3)
    receipts = ledger.list_inputs(job["job_id"])
    close = {
        **_common(),
        "idempotencyKey": "close-1",
        "jobId": job["job_id"],
        "clientExecutionId": job["client_execution_id"],
        "fencingToken": "1",
        "payloadCount": 1,
        "totalRows": 3,
        "totalBytes": receipts[0]["bytes"],
        "manifestSha256": manifest_sha256(receipts),
    }

    result = _dispatch(coordinator, INPUT_CLOSE_ACTION, close)

    assert result["input"]["state"] == "CLOSED"
    assert result["execution"]["state"] == "QUEUED"
    status = _dispatch(
        coordinator,
        STATUS_ACTION,
        {**_common(), "jobId": job["job_id"]},
    )
    assert status["input"]["state"] == "CLOSED"
    assert status["execution"]["state"] == "QUEUED"


def test_empty_fit_reports_empty_input_before_device_revalidation(
    tmp_path,
    postgres_ledger,
):
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        allow_plaintext=True,
    ).validate()
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    available = {"cuda": True}
    inventory = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            cuda_capacity=1 if available["cuda"] else 0,
        ),
    )
    coordinator = build_test_job_coordinator(
        config,
        postgres_ledger,
        spool,
        device_inventory=inventory,
    )
    create = {**_fit_create(), "device": "cuda"}
    _dispatch(coordinator, CREATE_ACTION, create)
    available["cuda"] = False
    close = {
        **_common(),
        "idempotencyKey": "close-empty-fit",
        "jobId": create["jobId"],
        "clientExecutionId": create["clientExecutionId"],
        "fencingToken": "1",
        "payloadCount": 0,
        "totalRows": 0,
        "totalBytes": 0,
        "manifestSha256": manifest_sha256([]),
    }

    with pytest.raises(ServiceError) as error:
        _dispatch(coordinator, INPUT_CLOSE_ACTION, close)

    assert error.value.code is ErrorCode.EMPTY_INPUT
    job = postgres_ledger.get_job(create["jobId"])
    assert job["input_state"] == "OPEN"
    assert job["execution_state"] == "WAITING_INPUT"


def test_close_replay_from_previous_owner_is_fenced_after_takeover(
    coordinator_components,
):
    _, _, ledger, coordinator = coordinator_components
    create = _fit_create()
    _dispatch(coordinator, CREATE_ACTION, create)
    job = ledger.get_job(create["jobId"])
    commit_input(ledger, job, 0)
    receipts = ledger.list_inputs(job["job_id"])
    close = {
        **_common(),
        "idempotencyKey": "close-before-takeover",
        "jobId": job["job_id"],
        "clientExecutionId": job["client_execution_id"],
        "fencingToken": "1",
        "payloadCount": 1,
        "totalRows": 1,
        "totalBytes": receipts[0]["bytes"],
        "manifestSha256": manifest_sha256(receipts),
    }
    _dispatch(coordinator, INPUT_CLOSE_ACTION, close)
    _dispatch(
        coordinator,
        ACQUIRE_ACTION,
        {
            **_common(),
            "idempotencyKey": "takeover-after-close",
            "jobId": job["job_id"],
            "previousClientExecutionId": job["client_execution_id"],
            "expectedFencingToken": "1",
            "clientExecutionId": str(uuid.uuid4()),
        },
    )

    with pytest.raises(ServiceError) as error:
        _dispatch(
            coordinator,
            INPUT_CLOSE_ACTION,
            {**close, "requestId": str(uuid.uuid4())},
        )

    assert error.value.code is ErrorCode.STALE_FENCE


def _publish_model(ledger, spool, *, label="daily"):
    fit = create_fit(ledger)
    commit_input(ledger, fit, 0, rows=1)
    close_input(ledger, fit)
    running = ledger.claim_execution_job(fit["job_id"], "cpu")
    model_ref = f"mdl_{uuid.uuid4().hex}"
    checkpoint = Path(spool.model_checkpoint_path(model_ref))
    spool.ensure_parent(str(checkpoint))
    checkpoint.write_bytes(b"certified checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata_path = spool.model_metadata_path(model_ref)
    spool.atomic_write_json(metadata_path, {"modelRef": model_ref})
    ledger.publish_model(
        fit["job_id"],
        running.attempt,
        attempt_id=running.attempt_id,
        model_ref=model_ref,
        label=label,
        generation=None,
        checkpoint_path=spool.model_relative_path(str(checkpoint)),
        metadata_path=spool.model_relative_path(metadata_path),
        byte_count=checkpoint.stat().st_size,
        sha256=digest,
        metadata={
            "model_config": model_config().to_dict(),
            "train_config": train_config().to_dict(),
            "data_contract": internal_data_contract(),
            "ml_contract": public_ml_contract(),
            "objective_config": objective_config(train_config()),
            "checkpoint": {"mlContract": public_ml_contract()},
        },
        result={"modelRef": model_ref},
    )
    return model_ref


def test_model_alias_is_owner_scoped_and_resolved_during_create(
    coordinator_components,
):
    _, spool, ledger, coordinator = coordinator_components
    model_ref = _publish_model(ledger, spool)
    create = {
        **_common(),
        "idempotencyKey": "predict-by-alias",
        "jobId": str(uuid.uuid4()),
        "clientExecutionId": str(uuid.uuid4()),
        "operation": "predict",
        "device": "cpu",
        "modelAlias": "daily",
        "predictionColumn": "forecast",
        "dataContract": public_data_contract(),
        "mlContract": public_ml_contract(),
    }

    result = _dispatch(coordinator, CREATE_ACTION, create)

    assert result["resolvedModelRef"] == model_ref
    assert ledger.get_job(create["jobId"])["resolved_model_ref"] == model_ref
    with pytest.raises(ServiceError) as error:
        _dispatch(
            coordinator,
            MODEL_DESCRIBE_ACTION,
            {**_common(), "modelAlias": "daily"},
            owner="another-owner",
        )
    assert error.value.code is ErrorCode.NOT_FOUND


def test_model_describe_has_stable_lifecycle_errors(coordinator_components):
    _, spool, ledger, coordinator = coordinator_components
    model_ref = _publish_model(ledger, spool)
    describe = {**_common(), "modelRef": model_ref}

    available = _dispatch(coordinator, MODEL_DESCRIBE_ACTION, describe)
    assert available["modelRef"] == model_ref
    assert available["generation"] == 1
    assert available["dataContract"]["dataContractSha256"] == (
        DATA_CONTRACT_SHA256
    )

    Path(spool.model_checkpoint_path(model_ref)).unlink()
    with pytest.raises(ServiceError) as missing:
        _dispatch(coordinator, MODEL_DESCRIBE_ACTION, describe)
    assert missing.value.code is ErrorCode.MODEL_UNAVAILABLE

    Path(spool.model_checkpoint_path(model_ref)).write_bytes(b"corrupt")
    with pytest.raises(ServiceError) as corrupt:
        _dispatch(coordinator, MODEL_DESCRIBE_ACTION, describe)
    assert corrupt.value.code is ErrorCode.MODEL_CORRUPT


def test_predict_contract_mismatch_is_rejected_before_job_creation(
    coordinator_components,
):
    _, spool, ledger, coordinator = coordinator_components
    model_ref = _publish_model(ledger, spool)
    create = {
        **_common(),
        "idempotencyKey": "predict-bad-contract",
        "jobId": str(uuid.uuid4()),
        "clientExecutionId": str(uuid.uuid4()),
        "operation": "predict",
        "device": "cpu",
        "modelRef": model_ref,
        "dataContract": public_data_contract(digest="f" * 64),
        "mlContract": public_ml_contract(),
    }

    with pytest.raises(ServiceError) as error:
        _dispatch(coordinator, CREATE_ACTION, create)

    assert error.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH
    assert ledger.get_job(create["jobId"]) is None
