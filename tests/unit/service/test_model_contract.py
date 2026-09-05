import pytest

from app.service.application.services.model_contract import verify_model_for_predict
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.service.domain.records import PublishedModelRecord
from tests.support.consumer_neutral import model_contract


def test_model_rejects_a_different_target_subset_as_schema_mismatch():
    stored_contract = model_contract(
        "multi-target-shared-resource",
        seq_len=2,
        feature_dim=3,
    )
    stored_document = stored_contract.to_document()
    data_contract = {
        "identity": "test.dataset",
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": "a" * 64,
        "seqLen": 2,
        "featureDim": 3,
    }
    stored_digests = stored_contract.digests("a" * 64)
    model = PublishedModelRecord(
        model_ref="mdl_generation",
        owner_subject="inventory",
        label="returns.daily",
        generation=1,
        checkpoint_path="mdl_generation/checkpoint.pth",
        byte_count=1024,
        sha256="b" * 64,
        metadata={
            "format": "transformer-checkpoint-v6",
            "dataContract": data_contract,
            "modelContract": stored_document,
            "semanticDigests": stored_digests,
            "initialization": {"kind": "random"},
        },
        data_contract=data_contract,
        model_contract=stored_document,
        semantic_digests=stored_digests,
        producing_job_id="00000000-0000-4000-8000-000000000001",
        created_at=1.0,
    )
    requested_contract = model_contract(
        "single-regression",
        seq_len=2,
        feature_dim=3,
    )

    with pytest.raises(ServiceError) as raised:
        verify_model_for_predict(
            model,
            model_contract=requested_contract.to_document(),
            semantic_digests=requested_contract.digests("a" * 64),
        )

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH


def test_model_uses_consumer_data_digest_not_opaque_envelope_for_compatibility():
    stored_contract = model_contract(
        "single-regression",
        seq_len=2,
        feature_dim=3,
    )
    stored_document = stored_contract.to_document()
    data_contract = {
        "identity": "test.dataset",
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": "a" * 64,
        "seqLen": 2,
        "featureDim": 3,
    }
    stored_digests = stored_contract.digests("a" * 64)
    model = PublishedModelRecord(
        model_ref="mdl_generation",
        owner_subject="inventory",
        label="returns.daily",
        generation=1,
        checkpoint_path="mdl_generation/checkpoint.pth",
        byte_count=1024,
        sha256="b" * 64,
        metadata={
            "format": "transformer-checkpoint-v6",
            "dataContract": data_contract,
            "modelContract": stored_document,
            "semanticDigests": stored_digests,
            "initialization": {"kind": "random"},
        },
        data_contract=data_contract,
        model_contract=stored_document,
        semantic_digests=stored_digests,
        producing_job_id="00000000-0000-4000-8000-000000000001",
        created_at=1.0,
    )

    resolved = verify_model_for_predict(
        model,
        model_contract=stored_document,
        semantic_digests=stored_digests,
    )

    assert resolved.seq_len == 2
    assert resolved.feature_dim == 3
