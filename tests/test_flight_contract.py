import json
import uuid

import pyarrow.flight as flight
import pytest

from app.flight.constants import (
    CONTRACT_NAME,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    MAX_MANIFEST_ITEMS,
    SEAL_ACTION,
)
from app.flight.contract import (
    canonical_request_hash,
    parse_action_body,
    parse_input_descriptor,
    parse_output_descriptor,
    validate_action_request,
    validate_upload_metadata,
)
from app.flight.errors import ServiceError


def request_id():
    return str(uuid.uuid4())


def fit_create_request(**overrides):
    document = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": request_id(),
        "idempotencyKey": "fit-create-1",
        "operation": "fit",
        "device": "cpu",
        "modelLabel": "returns.daily",
        "modelConfig": {"seqLen": 10},
        "trainingConfig": {"epochs": 2, "deterministic": True},
    }
    document.update(overrides)
    return document


def test_fit_create_contract_normalizes_and_validates_configs():
    parsed = validate_action_request(CREATE_ACTION, fit_create_request())

    assert parsed["operation"] == "fit"
    assert parsed["device"] == "cpu"
    assert parsed["model_label"] == "returns.daily"
    assert parsed["model_config"].seq_len == 10
    assert parsed["train_config"].epochs == 2
    assert parsed["train_config"].deterministic is True


def test_predict_create_accepts_one_opaque_model_selector():
    document = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": request_id(),
        "idempotencyKey": "predict-create-1",
        "operation": "predict",
        "device": "auto",
        "modelRef": "mdl_f4a62560",
        "predictionColumn": "out",
    }

    parsed = validate_action_request(CREATE_ACTION, document)

    assert parsed["model_ref"] == "mdl_f4a62560"
    assert parsed["model_selector"] == "modelRef"


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 2},
        {"device": "gpu"},
        {"modelLabel": "../../outside"},
        {"modelConfig": {"seqLen": 10, "argv": ["--epochs", "999"]}},
        {"trainingConfig": {"epochs": 2, "checkpointPath": "/tmp/a"}},
        {"unknown": True},
    ],
)
def test_create_rejects_invalid_version_paths_and_arbitrary_fields(overrides):
    with pytest.raises(ServiceError):
        validate_action_request(CREATE_ACTION, fit_create_request(**overrides))


def test_request_hash_ignores_retry_identifiers_but_detects_semantic_conflict():
    original = fit_create_request()
    retry = {
        **original,
        "requestId": request_id(),
        "idempotencyKey": "another-key",
    }
    conflict = {**retry, "device": "auto"}

    assert canonical_request_hash(original) == canonical_request_hash(retry)
    assert canonical_request_hash(original) != canonical_request_hash(conflict)


def test_seal_manifest_is_strict_and_language_neutral():
    payload_id = request_id()
    document = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": request_id(),
        "idempotencyKey": "seal-1",
        "jobId": request_id(),
        "manifest": [{
            "payloadId": payload_id,
            "ordinal": 0,
            "sha256": "a" * 64,
        }],
    }

    parsed = validate_action_request(SEAL_ACTION, document)

    assert parsed["manifest"][0] == {
        "payloadId": payload_id,
        "ordinal": 0,
        "sha256": "a" * 64,
    }


def test_seal_manifest_count_is_bounded_by_action_document_capacity():
    document = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": request_id(),
        "idempotencyKey": "seal-limit",
        "jobId": request_id(),
        "manifest": [
            {
                "payloadId": request_id(),
                "ordinal": index,
                "sha256": "a" * 64,
            }
            for index in range(MAX_MANIFEST_ITEMS + 1)
        ],
    }

    with pytest.raises(ServiceError, match="at most 400"):
        validate_action_request(SEAL_ACTION, document)


def test_upload_metadata_and_descriptors_are_strict():
    job_id = request_id()
    payload_id = request_id()
    metadata = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "jobId": job_id,
        "payloadId": payload_id,
        "ordinal": 3,
        "schemaId": FIT_SCHEMA_ID,
        "rows": 255,
    }

    parsed = validate_upload_metadata(metadata)
    assert parsed["rows"] == 255
    assert parse_input_descriptor(
        flight.FlightDescriptor.for_path(
            "transformer", "v1", "jobs", job_id, "inputs", "3"
        )
    ) == (job_id, 3)
    assert parse_output_descriptor(
        flight.FlightDescriptor.for_path(
            "transformer", "v1", "jobs", job_id, "outputs", "3"
        )
    ) == (job_id, 3)

    with pytest.raises(ServiceError):
        parse_input_descriptor(
            flight.FlightDescriptor.for_path(
                "transformer", "v1", "jobs", job_id, "inputs", "../3"
            )
        )
    with pytest.raises(ServiceError, match="non-negative integer"):
        parse_input_descriptor(
            flight.FlightDescriptor.for_path(
                "transformer", "v1", "jobs", job_id, "inputs", "9" * 5000
            )
        )


def test_action_body_rejects_non_json_and_oversized_documents():
    with pytest.raises(ServiceError, match="valid UTF-8 JSON"):
        parse_action_body(b"not-json")
    with pytest.raises(ServiceError, match="exceeds"):
        parse_action_body(b" " * (64 * 1024 + 1))
    with pytest.raises(ServiceError, match="valid UTF-8 JSON"):
        parse_action_body(b'{"ordinal":' + (b"9" * 5000) + b"}")


def test_action_body_accepts_utf8_json_object():
    document = fit_create_request()
    assert parse_action_body(json.dumps(document).encode()) == document
