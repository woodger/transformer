from dataclasses import replace

from app.service.application.queries.status import GetJobStatus
from app.service.domain.job import JobState
from app.service.domain.records import (
    InputRecord,
    JobRecord,
    OutputRecord,
    StatusRecoveryRecord,
    StatusSnapshot,
    TrainingRecoveryCheckpointRecord,
)


def test_status_query_maps_typed_snapshot_without_exposing_storage_paths():
    job_id = "00000000-0000-4000-8000-000000000001"
    snapshot = StatusSnapshot(
        job=JobRecord(
            job_id=job_id,
            owner_subject="inventory",
            operation="fit",
            state=JobState.SUCCEEDED,
            revision=8,
            requested_device="auto",
            selected_device="cuda",
            prediction_column="predictions",
            progress={"epoch": 1},
            attempt=2,
            error_code=None,
            error_message=None,
            result={"modelRef": "mdl_generation"},
            created_at=1.0,
            updated_at=8.0,
            sealed_at=2.0,
            queued_at=3.0,
            started_at=4.0,
            cancel_requested_at=None,
            finished_at=8.0,
        ),
        inputs=(InputRecord(
            job_id=job_id,
            ordinal=0,
            payload_id="00000000-0000-4000-8000-000000000002",
            schema_id="inventory.sequence.fit.v1",
            rows=4,
            batches=2,
            byte_count=1024,
            sha256="a" * 64,
            schema_fingerprint="b" * 64,
            relative_path="private/input.arrow",
            storage_class="recovery",
            source_width=4,
            feature_dim=2,
            committed_at=2.0,
        ),),
        outputs=(),
        recovery=StatusRecoveryRecord(
            checkpoint=TrainingRecoveryCheckpointRecord(
                job_id=job_id,
                generation=1,
                attempt=1,
                format="transformer-training-recovery-v1",
                relative_path="private/checkpoint.pth",
                byte_count=4096,
                sha256="e" * 64,
                completed_epochs=1,
                global_step=2,
                training_complete=False,
            ),
            retry_count=1,
            last_retry_code="EXECUTION_INTERRUPTED",
            resumed_from_generation=1,
        ),
    )

    class Ledger:
        def __init__(self, current):
            self.current = current

        def get_status_snapshot_record(self, requested_job_id, owner):
            assert (requested_job_id, owner) == (job_id, "inventory")
            return self.current

    query = GetJobStatus(
        Ledger(snapshot),
        path_version="v2",
        response_factory=lambda request_id, **body: {
            "requestId": request_id,
            **body,
        },
    )

    result = query.execute("inventory", job_id, "request-1")

    assert result["state"] == "SUCCEEDED"
    assert result["committedInputs"] == [{
        "payloadId": "00000000-0000-4000-8000-000000000002",
        "ordinal": 0,
        "schemaId": "inventory.sequence.fit.v1",
        "rows": 4,
        "batches": 2,
        "bytes": 1024,
        "sha256": "a" * 64,
        "schemaFingerprint": "b" * 64,
    }]
    assert result["results"] == {
        "outputs": [],
        "modelRef": "mdl_generation",
        "checkpoint": None,
    }
    assert result["recovery"]["latestCheckpoint"]["generation"] == 1
    assert "private" not in repr(result)

    prediction_snapshot = StatusSnapshot(
        job=replace(
            snapshot.job,
            operation="predict",
            result={},
        ),
        inputs=(replace(
            snapshot.inputs[0],
            schema_id="inventory.sequence.predict.v1",
            storage_class="runtime",
        ),),
        outputs=(OutputRecord(
            job_id=job_id,
            ordinal=0,
            rows=4,
            batches=1,
            byte_count=512,
            sha256="c" * 64,
            schema_fingerprint="d" * 64,
            relative_path="private/output.arrow",
            published_at=8.0,
        ),),
        recovery=None,
    )
    prediction_query = GetJobStatus(
        Ledger(prediction_snapshot),
        path_version="v2",
        response_factory=lambda request_id, **body: {
            "requestId": request_id,
            **body,
        },
    )

    prediction = prediction_query.execute(
        "inventory",
        job_id,
        "request-2",
    )

    assert prediction["results"] == {
        "outputs": [{
            "ordinal": 0,
            "descriptorPath": [
                "transformer",
                "v2",
                "jobs",
                job_id,
                "outputs",
                "0",
            ],
            "rows": 4,
            "bytes": 512,
        }],
        "modelRef": None,
        "checkpoint": None,
    }
    assert prediction["recovery"] is None
    assert "private" not in repr(prediction)
