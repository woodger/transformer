import pytest

from app.contracts.worker.v8.config import ModelConfig, TrainConfig
from app.contracts.worker.v8.objective import (
    ObjectiveConfig,
    default_objective,
    ml_contract,
    objective_config_sha256,
)
from app.service.application.services.model_contract import verify_model_semantics
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.service.domain.records import PublishedModelRecord


def test_model_rejects_a_different_target_subset_as_schema_mismatch():
    stored_objective = default_objective()
    stored_contract = ml_contract(stored_objective)
    model_config = ModelConfig(seq_len=2, feature_dim=3)
    train_config = TrainConfig()
    data_contract = {"data_contract_sha256": "a" * 64}
    model = PublishedModelRecord(
        model_ref="mdl_generation",
        owner_subject="inventory",
        label="returns.daily",
        generation=1,
        checkpoint_path="mdl_generation/checkpoint.pth",
        metadata_path="mdl_generation/metadata.json",
        byte_count=1024,
        sha256="b" * 64,
        metadata={
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "data_contract": data_contract,
            "ml_contract": stored_contract,
            "objective": stored_objective.to_document(),
            "checkpoint": {"mlContract": stored_contract},
        },
        data_contract=data_contract,
        ml_contract=stored_contract,
        objective_config_sha256=objective_config_sha256(stored_objective),
        producing_job_id="00000000-0000-4000-8000-000000000001",
        created_at=1.0,
    )
    requested_objective = ObjectiveConfig.from_document({
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

    with pytest.raises(ServiceError) as raised:
        verify_model_semantics(
            model,
            requested_ml_contract=ml_contract(requested_objective),
        )

    assert raised.value.code is ErrorCode.MODEL_SCHEMA_MISMATCH
