from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.contracts.worker.v11.config import ModelConfig, TrainConfig
from app.contracts.worker.v11.objective import (
    ObjectiveConfig,
    default_objective,
    ml_contract,
    objective_config_sha256,
)
from app.service.application.commands.jobs import CreateJobAction
from app.service.application.messages.jobs import (
    CreateJobCommand,
    ServiceLimits,
)
from app.service.application.ports.job_lifecycle import LifecycleMutation
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.service.domain.records import PublishedModelRecord


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
        model_verifier=SimpleNamespace(verify=verified.append),
        metrics=SimpleNamespace(
            add=lambda *_args: None,
            record_transition=lambda *_args: None,
        ),
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
    )

    result = action.create(command)

    assert result.initialization == {
        "kind": "publishedModel",
        "parentModelRef": parent.model_ref,
        "parentCheckpointSha256": parent.sha256,
        "parentDataContractSha256": "a" * 64,
        "dataContractSha256": "a" * 64,
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
        data_contract={
            **command.data_contract,
            "data_contract_sha256": "d" * 64,
        },
    )
    action, _, _ = _action_for_parent(parent)

    with pytest.raises(ServiceError) as raised:
        action.create(command)

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH
    assert raised.value.message == (
        "parent model data contract is not compatible with fit job"
    )


def test_predict_still_rejects_a_new_data_contract_digest():
    command, parent = _published_model_command()
    command = replace(
        command,
        operation="predict",
        data_contract={
            **command.data_contract,
            "data_contract_sha256": "d" * 64,
        },
        model_label=None,
        initialization_kind=None,
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
    single_target = ObjectiveConfig.from_document({
        "targets": ["MeanReturn"],
        "objective": {
            "schemaVersion": 1,
            "aggregation": "WeightedSum",
            "reduction": "GlobalRowMean",
            "directLosses": [
                {
                    "target": "MeanReturn",
                    "operator": "SmoothL1",
                    "weight": 1.0,
                }
            ],
            "auxiliaryLosses": [],
            "balancing": {"operator": "Static"},
        },
    })
    command = replace(
        command,
        data_contract={
            **command.data_contract,
            "data_contract_sha256": "d" * 64,
        },
        ml_contract=ml_contract(single_target),
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
    objective = default_objective()
    model_contract = ml_contract(objective)
    model_config = ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        feature_dim=2,
    )
    train_config = TrainConfig()
    data_contract = {
        "id": "inventory.learning-dataset",
        "version": 2,
        "profile": "research-dividend-events-v2",
        "data_contract_sha256": "a" * 64,
        "seq_len": 2,
        "feature_dim": 2,
        "target_schema_id": "inventory.target.v2",
    }
    parent = PublishedModelRecord(
        model_ref="mdl_parent",
        owner_subject="inventory",
        label="returns.daily",
        generation=1,
        checkpoint_path="mdl_parent/checkpoint.pth",
        byte_count=1024,
        sha256="b" * 64,
        metadata={
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "data_contract": data_contract,
            "ml_contract": model_contract,
            "objective": objective.to_document(),
            "checkpoint": {"mlContract": model_contract},
        },
        data_contract=data_contract,
        ml_contract=model_contract,
        objective_config_sha256=objective_config_sha256(objective),
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
            "kind": "indexedFeatureBlocks",
            "featureBlocks": [
                {"position": 0, "windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        data_contract=data_contract,
        ml_contract=model_contract,
        model_label="returns.daily.fine-tuned",
        model_selector="reference",
        model_ref=parent.model_ref,
        model_config=model_config,
        training_config=train_config,
        initialization_kind="publishedModel",
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
