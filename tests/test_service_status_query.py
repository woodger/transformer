from dataclasses import replace

import pytest

from app.service.adapters.inbound.flight.presentation import present_job_status
from app.service.application.job_models import GetJobStatusQuery
from app.service.application.queries.status import GetJobStatus
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import (
    JobRecord,
    StatusRecoveryRecord,
    StatusSnapshot,
    TrainingRecoveryCheckpointRecord,
)

JOB_ID = "00000000-0000-4000-8000-000000000001"


def _job(**overrides):
    value = JobRecord(
        job_id=JOB_ID,
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.CLOSED,
        execution_state=ExecutionState.SUCCEEDED,
        revision=8,
        input_revision=3,
        next_input_ordinal=3,
        payload_count=3,
        total_rows=12,
        total_bytes=4096,
        manifest_sha256="a" * 64,
        client_execution_id="00000000-0000-4000-8000-000000000002",
        fencing_token=7,
        requested_device="auto",
        selected_device="cuda",
        resolved_model_ref=None,
        prediction_column="predictions",
        data_contract={
            "id": "inventory.learning-dataset",
            "version": 1,
            "data_contract_sha256": "b" * 64,
            "seq_len": 2,
            "feature_dim": 2,
            "target_schema_id": "inventory.target.v1",
        },
        ml_contract={
            "targetSchemaId": "inventory.target.v1",
            "objectiveId": "transformer.objective.target-aligned.v1",
        },
        progress={"epoch": 2},
        attempt=2,
        error_code=None,
        error_message=None,
        result={"modelRef": "mdl_generation"},
        created_at=1.0,
        updated_at=8.0,
        input_closed_at=2.0,
        queued_at=3.0,
        started_at=4.0,
        cancel_requested_at=None,
        finished_at=8.0,
    )
    return replace(value, **overrides)


class Ledger:
    def __init__(self, snapshot, identity=None):
        self.snapshot = snapshot
        self.identity = identity

    def get_status_snapshot_record(self, job_id, owner):
        assert (job_id, owner) == (JOB_ID, "inventory")
        return self.snapshot

    def is_job_retired(self, job_id, *, owner_subject):
        assert (job_id, owner_subject) == (JOB_ID, "inventory")
        return self.identity is not None


def _query(snapshot, identity=None):
    return GetJobStatus(Ledger(snapshot, identity))


def _execute(query, request_id):
    return present_job_status(
        query.execute(
            GetJobStatusQuery(
                owner_subject="inventory",
                request_id=request_id,
                job_id=JOB_ID,
            )
        )
    )


def test_status_exposes_bounded_v4_state_without_artifact_paths():
    recovery = StatusRecoveryRecord(
        checkpoint=TrainingRecoveryCheckpointRecord(
            job_id=JOB_ID,
            generation=2,
            attempt=1,
            format="transformer-training-recovery-v3",
            relative_path="private/checkpoint.pth",
            byte_count=4096,
            sha256="c" * 64,
            completed_epochs=2,
            global_step=6,
            training_complete=False,
        ),
        retry_count=1,
        last_retry_code="EXECUTION_INTERRUPTED",
        resumed_from_generation=2,
    )
    snapshot = StatusSnapshot(
        job=_job(),
        output_count=0,
        recovery=recovery,
    )

    result = _execute(_query(snapshot), "request-1")

    assert result["input"] == {
        "state": "CLOSED",
        "revision": 3,
        "nextOrdinal": 3,
        "payloadCount": 3,
        "totalRows": 12,
        "totalBytes": 4096,
        "manifestSha256": "a" * 64,
    }
    assert result["execution"] == {"state": "SUCCEEDED", "attempt": 2}
    assert result["ownership"]["fencingToken"] == "7"
    assert result["results"] == {
        "outputCount": 0,
        "modelRef": "mdl_generation",
        "checkpoint": None,
    }
    assert result["recovery"]["latestCheckpoint"]["generation"] == 2
    assert result["pollAfterMs"] == 0
    assert "private" not in repr(result)


def test_failed_status_has_stable_error_and_nonterminal_status_polls():
    failed = StatusSnapshot(
        job=_job(
            execution_state=ExecutionState.FAILED,
            error_code=ErrorCode.INPUT_TIMEOUT.value,
            error_message="input timed out",
            result=None,
        ),
        output_count=0,
        recovery=None,
    )
    failure = _execute(_query(failed), "request-2")
    assert failure["error"] == {
        "code": "INPUT_TIMEOUT",
        "message": "input timed out",
    }

    running = StatusSnapshot(
        job=_job(
            input_state=InputState.OPEN,
            execution_state=ExecutionState.RUNNING,
            manifest_sha256=None,
            input_closed_at=None,
            finished_at=None,
            result=None,
        ),
        output_count=0,
        recovery=None,
    )
    active = _execute(_query(running), "request-3")
    assert active["input"]["state"] == "OPEN"
    assert active["execution"]["state"] == "RUNNING"
    assert active["pollAfterMs"] == 500


def test_retired_identity_returns_stable_job_retired_error():
    snapshot = StatusSnapshot(job=None, output_count=0, recovery=None)
    query = _query(snapshot, {"retired_at": 10.0})

    with pytest.raises(ServiceError) as error:
        query.execute(GetJobStatusQuery(
            owner_subject="inventory",
            request_id="request-4",
            job_id=JOB_ID,
        ))

    assert error.value.code is ErrorCode.JOB_RETIRED
