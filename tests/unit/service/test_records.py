from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v16.config import ModelConfig, TrainConfig
from app.contracts.worker.v16.model_definition import resolved_semantic_digests
from app.service.adapters.outbound.postgres.config import DatabaseConfig
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.mapping import (
    execution_job_from_mapping,
    recoverable_attempt_from_mapping,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import (
    CommittedInputRecord,
    ExecutionJobRecord,
    RecoverableAttemptRecord,
)
from tests.fixture_documents import semantic_fixture_document

MODEL_CONTRACT = ModelContract.from_document(
    semantic_fixture_document("single-regression")["modelContract"],
)
MODEL_CONTRACT_DOCUMENT = MODEL_CONTRACT.to_document()
SEMANTIC_DIGESTS = resolved_semantic_digests(
    MODEL_CONTRACT,
    "a" * 64,
    ModelConfig.from_tuning(
        MODEL_CONTRACT.model_tuning,
        seq_len=2,
        feature_dim=2,
    ),
)
DATA_CONTRACT = {
    "identity": "test.dataset",
    "revision": 1,
    "profile": "test.profile",
    "dataContractSha256": "a" * 64,
    "seqLen": 2,
    "featureDim": 2,
}


def _assert_frozen_slots(record):
    assert not hasattr(record, "__dict__")
    field = fields(record)[0]
    with pytest.raises(FrozenInstanceError):
        setattr(record, field.name, getattr(record, field.name))


def test_execution_mapping_preserves_both_state_axes_and_typed_config():
    train_config = TrainConfig(epochs=3, deterministic=True)
    value = {
        "job_id": "00000000-0000-4000-8000-000000000001",
        "owner_subject": "inventory",
        "operation": "fit",
        "input_state": "OPEN",
        "execution_state": "RUNNING",
        "input_revision": 3,
        "selected_device": "cuda",
        "model_label": "daily",
        "resolved_model_ref": None,
        "initialization": {"source": "random"},
        "prediction_column": "out",
        "source_encoding": {
            "featureBlocks": [
                {"windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        "model_config": ModelConfig(
            seq_len=2,
            feature_dim=2,
        ).to_dict(),
        "training_config": train_config.to_dict(),
        "data_contract": DATA_CONTRACT,
        "model_contract": MODEL_CONTRACT_DOCUMENT,
        "semantic_digests": SEMANTIC_DIGESTS,
        "config_hash": "b" * 64,
        "manifest_sha256": None,
        "feature_dim": 2,
        "next_input_ordinal": 2,
        "attempt": 1,
        "device_id": "GPU-opaque",
        "resume_generation": None,
        "queued_at": 1.0,
        "started_at": 2.0,
        "attempt_id": "00000000-0000-4000-8000-000000000002",
    }

    record = execution_job_from_mapping(value)

    assert isinstance(record, ExecutionJobRecord)
    assert record.input_state is InputState.OPEN
    assert record.execution_state is ExecutionState.RUNNING
    assert record.input_frame_count == 2
    assert record.model_config == ModelConfig(seq_len=2, feature_dim=2)
    assert record.training_config == TrainConfig(
        epochs=3,
        deterministic=True,
    )
    assert record.model_contract == MODEL_CONTRACT_DOCUMENT
    assert record.semantic_digests == SEMANTIC_DIGESTS
    _assert_frozen_slots(record)


def test_job_creation_persists_round_trippable_training_diagnostics():
    job_id = "00000000-0000-4000-8000-000000000001"
    train_config = TrainConfig()
    database = Database(
        DatabaseConfig(
            host="localhost",
            database="transformer",
            user="transformer",
            password="secret",
        ),
        engine=create_engine("sqlite+pysqlite:///:memory:"),
    )
    ledger = Ledger(database)
    session = cast(
        Session,
        SimpleNamespace(add=lambda _record: None, flush=lambda: None),
    )

    try:
        stored = ledger.create_job(
            job_id=job_id,
            owner_subject="inventory",
            client_execution_id="00000000-0000-4000-8000-000000000002",
            operation="fit",
            requested_device="cuda",
            prediction_column="out",
            config_hash="b" * 64,
            source_encoding={
                "featureBlocks": [
                    {"windowRows": 1, "nativeRowWidth": 2},
                ],
            },
            data_contract=DATA_CONTRACT,
            model_contract=MODEL_CONTRACT_DOCUMENT,
            semantic_digests=SEMANTIC_DIGESTS,
            create_result={"jobId": job_id},
            model_label="daily",
            model_config=ModelConfig(seq_len=2, feature_dim=2),
            training_config=train_config,
            initialization={"source": "random"},
            now=1.0,
            connection=session,
        )
    finally:
        database.close()

    assert stored["training_config"] == train_config.to_dict()
    assert TrainConfig.from_dict(stored["training_config"]) == train_config


def test_committed_input_record_carries_order_and_contract_identity():
    record = CommittedInputRecord(
        job_id="00000000-0000-4000-8000-000000000001",
        ordinal=4,
        payload_id="00000000-0000-4000-8000-000000000002",
        commit_revision=7,
        schema_id="transformer.indexed-feature-blocks.fit.v1",
        data_contract_sha256="a" * 64,
        chunks=1,
        rows=10,
        native_rows=(12,),
        first_range_ordinal=0,
        first_example_offset=0,
        last_range_ordinal=0,
        next_example_offset=10,
        batches=2,
        byte_count=4096,
        sha256="b" * 64,
        schema_fingerprint="c" * 64,
        relative_path="jobs/id/inputs/candidate.arrow",
        storage_class="recovery",
    )

    assert (record.ordinal, record.commit_revision) == (4, 7)
    assert record.data_contract_sha256 == "a" * 64
    _assert_frozen_slots(record)


def test_recoverable_attempt_mapping_keeps_internal_attempt_identity():
    value = {
        "job_id": "00000000-0000-4000-8000-000000000001",
        "attempt": 2,
        "attempt_id": "00000000-0000-4000-8000-000000000002",
        "pid": 12001,
        "pgid": 12002,
        "boot_id": "00000000-0000-4000-8000-000000000003",
        "process_start_ticks": 987654,
    }

    record = recoverable_attempt_from_mapping(value)

    assert record == RecoverableAttemptRecord(**value)
    _assert_frozen_slots(record)
