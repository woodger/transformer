from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.contracts.flight.v14 import job_config_sha256
from app.contracts.semantic.v2 import ModelContract
from app.contracts.worker.v13.config import ModelConfig, TrainConfig
from app.service.application.commands.jobs import CreateJobAction
from app.service.application.messages.jobs import (
    CreateJobCommand,
    ServiceLimits,
)
from app.service.application.ports.job_lifecycle import LifecycleMutation
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.service.domain.records import PublishedModelRecord
from tests.support.consumer_neutral import model_contract


def test_published_model_fit_resolves_immutable_parent_lineage():
    command, parent = _published_model_command()
    verified = []
    prepared = []

    def create(command, *, max_active_jobs, preflight, prepare):
        assert max_active_jobs == 2
        preflight()
        result = prepare(parent)
        prepared.append(result)
        return LifecycleMutation(result.result, replayed=False)

    action = CreateJobAction(
        SimpleNamespace(create=create),
        max_active_jobs=2,
        limits=_limits(),
        cuda_available=lambda: False,
        is_draining=lambda: False,
        job_config_digest=job_config_sha256,
        model_verifier=SimpleNamespace(verify=verified.append),
        metrics=SimpleNamespace(
            add=lambda *_args: None,
            record_transition=lambda *_args: None,
        ),
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
    )

    result = action.create(command)

    assert result.initialization == {
        "source": "publishedModel",
        "parentModelRef": parent.model_ref,
        "parentCheckpointSha256": parent.sha256,
        "parentDataContractSha256": "a" * 64,
        "dataContractSha256": "a" * 64,
        "parentTargetContractSha256": parent.semantic_digests[
            "targetContractSha256"
        ],
        "targetContractSha256": parent.semantic_digests[
            "targetContractSha256"
        ],
        "parentObjectiveSha256": parent.semantic_digests["objectiveSha256"],
        "objectiveSha256": parent.semantic_digests["objectiveSha256"],
        "parentModelContractSha256": parent.semantic_digests[
            "modelContractSha256"
        ],
        "modelContractSha256": parent.semantic_digests[
            "modelContractSha256"
        ],
    }
    assert result.resolved_model_ref == parent.model_ref
    assert prepared[0].resolved_model_ref == parent.model_ref
    assert verified == [parent]


def test_published_model_fit_rejects_a_different_model_configuration():
    command, parent = _published_model_command()
    command = replace(
        command,
        model_config=ModelConfig(
            seq_len=2,
            hidden=16,
            layers=1,
            dropout=0.0,
            nhead=2,
            feature_dim=2,
        ),
    )

    def create(_command, *, max_active_jobs, preflight, prepare):
        assert max_active_jobs == 2
        preflight()
        return LifecycleMutation(prepare(parent).result, replayed=False)

    action = CreateJobAction(
        SimpleNamespace(create=create),
        max_active_jobs=2,
        limits=_limits(),
        cuda_available=lambda: False,
        is_draining=lambda: False,
        job_config_digest=job_config_sha256,
        model_verifier=SimpleNamespace(verify=lambda _model: None),
        metrics=SimpleNamespace(
            add=lambda *_args: None,
            record_transition=lambda *_args: None,
        ),
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
    )

    with pytest.raises(ServiceError) as raised:
        action.create(command)

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH
    assert raised.value.message == (
        "parent model configuration does not match fit job"
    )


def test_published_model_fit_rejects_a_new_data_contract_digest():
    command, parent = _published_model_command()
    command = replace(
        command,
        data_contract={**command.data_contract, "dataContractSha256": "d" * 64},
        semantic_digests={
            **command.semantic_digests,
            "dataContractSha256": "d" * 64,
        },
    )
    action, _, _ = _action_for_parent(parent)

    with pytest.raises(ServiceError) as raised:
        action.create(command)

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH
    assert raised.value.message == (
        "model data contract does not match the requested job"
    )


def test_published_model_fit_treats_data_envelope_as_opaque():
    command, parent = _published_model_command()
    command = replace(
        command,
        data_contract={
            **command.data_contract,
            "identity": "test.opaque-envelope-renamed",
        },
    )
    action, prepared, _ = _action_for_parent(parent)

    result = action.create(command)

    assert result.semantic_digests == parent.semantic_digests
    assert prepared[0].result.data_contract["identity"] == (
        "test.opaque-envelope-renamed"
    )


def test_predict_still_rejects_a_new_data_contract_digest():
    command, parent = _published_model_command()
    command = replace(
        command,
        operation="predict",
        data_contract={**command.data_contract, "dataContractSha256": "d" * 64},
        semantic_digests={
            **command.semantic_digests,
            "dataContractSha256": "d" * 64,
        },
        model_label=None,
        initialization_source=None,
    )
    action, _, _ = _action_for_parent(parent)

    with pytest.raises(ServiceError) as raised:
        action.create(command)

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH
    assert raised.value.message == (
        "model data contract does not match the requested job"
    )


def test_published_model_fit_rejects_a_different_objective():
    command, parent = _published_model_command()
    changed_document = deepcopy(command.model_contract)
    changed_document["objective"]["directComponents"][0]["weight"] = 0.5
    changed_contract = ModelContract.from_document(changed_document)
    changed_digests = changed_contract.digests("a" * 64)
    command = replace(
        command,
        model_contract=changed_contract.to_document(),
        semantic_digests=changed_digests,
    )
    action, _, _ = _action_for_parent(parent)

    with pytest.raises(ServiceError) as raised:
        action.create(command)

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH


def _action_for_parent(parent):
    verified = []
    prepared = []

    def create(_command, *, max_active_jobs, preflight, prepare):
        assert max_active_jobs == 2
        preflight()
        result = prepare(parent)
        prepared.append(result)
        return LifecycleMutation(result.result, replayed=False)

    action = CreateJobAction(
        SimpleNamespace(create=create),
        max_active_jobs=2,
        limits=_limits(),
        cuda_available=lambda: False,
        is_draining=lambda: False,
        job_config_digest=job_config_sha256,
        model_verifier=SimpleNamespace(verify=verified.append),
        metrics=SimpleNamespace(
            add=lambda *_args: None,
            record_transition=lambda *_args: None,
        ),
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
    )
    return action, prepared, verified


def _published_model_command(
) -> tuple[CreateJobCommand, PublishedModelRecord]:
    semantic_contract = model_contract(
        "single-regression",
        seq_len=2,
        feature_dim=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
    )
    model_config = ModelConfig.from_manifest(semantic_contract.model_config)
    train_config = TrainConfig()
    data_contract = {
        "identity": "test.dataset",
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": "a" * 64,
        "seqLen": 2,
        "featureDim": 2,
    }
    model_contract_document = semantic_contract.to_document()
    semantic_digests = semantic_contract.digests("a" * 64)
    parent = PublishedModelRecord(
        model_ref="mdl_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        owner_subject="inventory",
        label="returns.daily",
        generation=1,
        checkpoint_path="mdl_parent/checkpoint.pth",
        byte_count=1024,
        sha256="b" * 64,
        metadata={
        "format": "transformer-checkpoint-v7",
            "dataContract": data_contract,
            "modelContract": model_contract_document,
            "semanticDigests": semantic_digests,
            "initialization": {"source": "random"},
        },
        data_contract=data_contract,
        model_contract=model_contract_document,
        semantic_digests=semantic_digests,
        producing_job_id="00000000-0000-4000-8000-000000000001",
        created_at=1.0,
    )
    command = CreateJobCommand(
        owner_subject="inventory",
        request_id="00000000-0000-4000-8000-000000000002",
        idempotency_key="fit:published-model",
        request_hash="c" * 64,
        job_id="00000000-0000-4000-8000-000000000003",
        client_execution_id="00000000-0000-4000-8000-000000000004",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        source_encoding={
            "encoding": "indexedFeatureBlocks",
            "featureBlocks": [
                {"position": 0, "windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        data_contract=data_contract,
        model_contract=model_contract_document,
        semantic_digests=semantic_digests,
        model_label="returns.daily.fine-tuned",
        model_selector="reference",
        model_ref=parent.model_ref,
        model_config=model_config,
        training_config=train_config,
        initialization_source="publishedModel",
    )
    return command, parent


def _limits() -> ServiceLimits:
    return ServiceLimits(
        max_message_bytes=1024,
        target_batch_bytes=512,
        max_batch_bytes=1024,
        max_payload_bytes=2048,
        max_rows_per_payload=100,
        max_payloads_per_job=10,
        max_job_bytes=4096,
        max_active_jobs_per_subject=2,
        max_page_items=100,
        input_idle_timeout_seconds=30.0,
    )
