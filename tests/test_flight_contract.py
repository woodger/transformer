from __future__ import annotations

import json
import uuid

import pyarrow.flight as flight
import pytest

from app.flight.constants import (
    ACQUIRE_ACTION,
    CREATE_ACTION,
    INPUTS_LIST_ACTION,
    MAX_PAGE_ITEMS,
)
from app.flight.contract import (
    canonical_request_hash,
    parse_action_body,
    parse_input_descriptor,
    parse_output_descriptor,
    validate_action_request,
    validate_upload_metadata,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode

REQUEST_ID = "11111111-1111-4111-8111-111111111111"
JOB_ID = "22222222-2222-4222-8222-222222222222"
EXECUTION_ID = "33333333-3333-4333-8333-333333333333"
SHA256 = "a" * 64


def _common(**fields) -> dict:
    return {
        "contract": "transformer-flight",
        "version": 3,
        "requestId": REQUEST_ID,
        **fields,
    }


def _fit_create(**fields) -> dict:
    document = _common(
        idempotencyKey="fit-create-1",
        jobId=JOB_ID,
        clientExecutionId=EXECUTION_ID,
        operation="fit",
        device="cpu",
        modelLabel="daily",
        modelConfig={"seqLen": 2},
        dataContract={
            "id": "inventory.learning-dataset",
            "version": 1,
            "dataContractSha256": SHA256,
            "seqLen": 2,
            "featureDim": 4,
            "targetSchemaId": "inventory.target.v1",
        },
    )
    document.update(fields)
    return document


def test_create_requires_client_identity_and_semantic_data_contract():
    parsed = validate_action_request(CREATE_ACTION, _fit_create())

    assert parsed["job_id"] == JOB_ID
    assert parsed["client_execution_id"] == EXECUTION_ID
    assert parsed["data_contract"] == {
        "id": "inventory.learning-dataset",
        "version": 1,
        "data_contract_sha256": SHA256,
        "seq_len": 2,
        "feature_dim": 4,
        "target_schema_id": "inventory.target.v1",
    }
    assert parsed["model_config"].feature_dim == 4

    missing = _fit_create()
    del missing["dataContract"]
    with pytest.raises(ServiceError) as error:
        validate_action_request(CREATE_ACTION, missing)
    assert error.value.code is ErrorCode.INVALID_ARGUMENT


def test_predict_accepts_exactly_one_owner_scoped_model_selector():
    request = _fit_create(
        operation="predict",
        modelAlias="daily",
        predictionColumn="forecast",
    )
    for key in ("modelLabel", "modelConfig"):
        request.pop(key)
    parsed = validate_action_request(CREATE_ACTION, request)
    assert parsed["model_selector"] == "modelAlias"
    assert parsed["model_ref"] == "daily"

    request["modelRef"] = "mdl_generation"
    with pytest.raises(ServiceError, match="exactly one"):
        validate_action_request(CREATE_ACTION, request)


def test_fencing_tokens_are_positive_canonical_decimal_strings():
    request = _common(
        idempotencyKey="acquire-1",
        jobId=JOB_ID,
        previousClientExecutionId=EXECUTION_ID,
        expectedFencingToken="7",
        clientExecutionId=str(uuid.uuid4()),
    )
    assert validate_action_request(ACQUIRE_ACTION, request)[
        "expected_fencing_token"
    ] == 7

    for invalid in (7, "0", "01", "+1", str(2**63)):
        request["expectedFencingToken"] = invalid
        with pytest.raises(ServiceError, match="FencingToken"):
            validate_action_request(ACQUIRE_ACTION, request)


def test_revision_pagination_requires_one_stable_snapshot_pair():
    first = _common(jobId=JOB_ID, afterRevision=4, limit=MAX_PAGE_ITEMS)
    parsed = validate_action_request(INPUTS_LIST_ACTION, first)
    assert parsed["snapshot_revision"] is None
    assert parsed["cursor"] is None

    later = _common(
        jobId=JOB_ID,
        afterRevision=4,
        snapshotRevision=9,
        cursor=6,
        limit=1,
    )
    parsed = validate_action_request(INPUTS_LIST_ACTION, later)
    assert (parsed["after_revision"], parsed["cursor"], parsed["snapshot_revision"]) == (
        4,
        6,
        9,
    )

    del later["cursor"]
    with pytest.raises(ServiceError, match="supplied together"):
        validate_action_request(INPUTS_LIST_ACTION, later)


def test_upload_metadata_has_no_request_id_and_is_fenced():
    metadata = {
        "contract": "transformer-flight",
        "version": 3,
        "jobId": JOB_ID,
        "clientExecutionId": EXECUTION_ID,
        "fencingToken": "8",
        "payloadId": str(uuid.uuid4()),
        "ordinal": 0,
        "schemaId": "inventory.sequence.fit.v2",
        "dataContractSha256": SHA256,
        "rows": 10,
    }
    parsed = validate_upload_metadata(metadata)
    assert parsed["fencing_token"] == 8

    metadata["requestId"] = REQUEST_ID
    with pytest.raises(ServiceError, match="must not contain requestId"):
        validate_upload_metadata(metadata)


def test_v3_input_and_output_descriptor_paths_are_strict():
    input_descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v3", "jobs", JOB_ID, "inputs", "12"
    )
    output_descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v3", "jobs", JOB_ID, "outputs", "3"
    )
    assert parse_input_descriptor(input_descriptor) == (JOB_ID, 12)
    assert parse_output_descriptor(output_descriptor) == (JOB_ID, 3)

    malformed = flight.FlightDescriptor.for_path(
        "transformer", "jobs", JOB_ID, "inputs", "0"
    )
    with pytest.raises(ServiceError, match="invalid input"):
        parse_input_descriptor(malformed)


def test_request_hash_ignores_transport_retry_identity_only():
    first = _fit_create()
    second = json.loads(json.dumps(first))
    second["requestId"] = str(uuid.uuid4())
    second["idempotencyKey"] = "fit-create-retry"
    assert canonical_request_hash(first) == canonical_request_hash(second)

    second["device"] = "cuda"
    assert canonical_request_hash(first) != canonical_request_hash(second)


def test_action_body_is_bounded_utf8_json_object():
    assert parse_action_body(json.dumps(_common()).encode()) == _common()
    for body in (b"[]", b"\xff", b"{" + b"x" * (64 * 1024)):
        with pytest.raises(ServiceError):
            parse_action_body(body)
