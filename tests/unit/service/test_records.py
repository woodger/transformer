from dataclasses import FrozenInstanceError, fields

import pytest

from app.contracts.worker.v6.config import ModelConfig, TrainConfig
from app.contracts.worker.v6.objective import ml_contract
from app.service.adapters.outbound.postgres.mapping import (
    execution_job_from_mapping,
    recoverable_attempt_from_mapping,
)
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import (
    CommittedInputRecord,
    ExecutionJobRecord,
    RecoverableAttemptRecord,
)


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
        "prediction_column": "out",
        "model_config": ModelConfig(
            seq_len=2,
            feature_dim=2,
        ).to_dict(),
        "training_config": train_config.to_dict(),
        "data_contract": {"data_contract_sha256": "a" * 64},
        "ml_contract": ml_contract(train_config),
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
    assert record.ml_contract == ml_contract(train_config)
    _assert_frozen_slots(record)


def test_committed_input_record_carries_order_and_contract_identity():
    record = CommittedInputRecord(
        job_id="00000000-0000-4000-8000-000000000001",
        ordinal=4,
        payload_id="00000000-0000-4000-8000-000000000002",
        commit_revision=7,
        schema_id="inventory.sequence.fit.v2",
        data_contract_sha256="a" * 64,
        rows=10,
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
