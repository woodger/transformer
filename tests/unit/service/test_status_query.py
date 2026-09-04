from dataclasses import replace

import pytest

from app.contracts.worker.v11.objective import default_objective, ml_contract
from app.service.adapters.inbound.flight.presentation import present_job_status
from app.service.application.messages.jobs import GetJobStatusQuery
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
        total_chunks=4,
        total_rows=12,
        total_native_rows=(15,),
        range_count=2,
        total_bytes=4096,
        manifest_sha256="a" * 64,
        client_execution_id="00000000-0000-4000-8000-000000000002",
        fencing_token=7,
        requested_device="auto",
        selected_device="cuda",
        resolved_model_ref=None,
        prediction_column="predictions",
        source_encoding={
            "kind": "indexedFeatureBlocks",
            "featureBlocks": [
                {"position": 0, "windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        data_contract={
            "id": "inventory.learning-dataset",
            "version": 2,
            "profile": "research-dividend-events-v2",
            "data_contract_sha256": "b" * 64,
            "seq_len": 2,
            "feature_dim": 2,
            "target_schema_id": "inventory.target.v2",
        },
        ml_contract=ml_contract(default_objective()),
        progress={
            "epoch": 2,
            "step": 6,
            "loss": -3.149016,
        },
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
        initialization={"kind": "random"},
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


def test_status_exposes_bounded_state_without_artifact_paths():
    recovery = StatusRecoveryRecord(
        checkpoint=TrainingRecoveryCheckpointRecord(
            job_id=JOB_ID,
            generation=2,
            attempt=1,
            format="transformer-training-recovery-v5",
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
        "totalChunks": 4,
        "totalLogicalRows": 12,
        "totalNativeRows": [15],
        "rangeCount": 2,
        "totalBytes": 4096,
        "manifestSha256": "a" * 64,
    }
    assert result["execution"] == {"state": "SUCCEEDED", "attempt": 2}
    assert result["ownership"]["fencingToken"] == "7"
    assert result["device"] == {"requested": "auto", "selected": "gpu"}
    assert result["progress"] == {
        "epoch": 2,
        "step": 6,
        "loss": -3.149016,
    }
    assert result["results"] == {
        "outputCount": 0,
        "modelRef": "mdl_generation",
        "checkpoint": None,
    }
    assert result["recovery"]["latestCheckpoint"]["generation"] == 2
    assert result["pollAfterMs"] == 0
    assert "private" not in repr(result)


def test_status_adds_implicit_random_lineage_to_existing_checkpoint_metadata():
    snapshot = StatusSnapshot(
        job=_job(result={
            "modelRef": "mdl_generation",
            "checkpoint": {
                "format": "transformer-checkpoint-v5",
                "sha256": "c" * 64,
            },
        }),
        output_count=0,
        recovery=None,
    )

    result = _execute(_query(snapshot), "request-existing-checkpoint")

    assert result["results"]["checkpoint"]["initialization"] == {
        "kind": "random"
    }


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


def test_status_maps_legacy_internal_cuda_oom_to_public_gpu_error():
    failed = StatusSnapshot(
        job=_job(
            execution_state=ExecutionState.FAILED,
            error_code="CUDA_OUT_OF_MEMORY",
            error_message="CUDA execution ran out of memory",
            result=None,
        ),
        output_count=0,
        recovery=None,
    )

    result = _execute(_query(failed), "request-gpu-oom")

    assert result["error"] == {
        "code": "GPU_OUT_OF_MEMORY",
        "message": "GPU execution ran out of memory",
    }


def test_status_hides_cuda_backend_name_from_public_error_message():
    failed = StatusSnapshot(
        job=_job(
            execution_state=ExecutionState.FAILED,
            error_code="DEVICE_LOST",
            error_message="CUDA device became unavailable during execution",
            result=None,
        ),
        output_count=0,
        recovery=None,
    )

    result = _execute(_query(failed), "request-device-lost")

    assert result["error"] == {
        "code": "DEVICE_LOST",
        "message": "GPU device became unavailable during execution",
    }


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
