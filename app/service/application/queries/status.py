from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.service.domain.errors import not_found
from app.service.domain.job import TERMINAL_STATES, JobState


class GetJobStatus:
    """Read one consistent durable status snapshot."""

    def __init__(
        self,
        ledger,
        *,
        path_version: str,
        response_factory: Callable[..., dict],
    ):
        self.ledger = ledger
        self.path_version = path_version
        self._response = response_factory

    def execute(self, owner: str, job_id: str, request_id: str) -> dict:
        snapshot = self.ledger.get_status_snapshot_record(
            job_id,
            owner,
        )
        job = snapshot.job
        if job is None:
            raise not_found("job not found")
        terminal = job.state in TERMINAL_STATES
        result = job.result or {}
        error = None
        if job.state == JobState.FAILED:
            error = {
                "code": job.error_code,
                "message": job.error_message,
            }
        return self._response(
            request_id,
            jobId=job_id,
            operation=job.operation,
            state=job.state.value,
            revision=job.revision,
            timestamps=_timestamps(job),
            device={
                "requested": job.requested_device,
                "selected": job.selected_device,
            },
            committedInputs=[_safe_input(item) for item in snapshot.inputs],
            progress=job.progress,
            attempt=job.attempt,
            recovery=_safe_recovery(snapshot.recovery),
            error=error,
            results={
                "outputs": [
                    {
                        "ordinal": item.ordinal,
                        "descriptorPath": [
                            "transformer",
                            self.path_version,
                            "jobs",
                            job_id,
                            "outputs",
                            str(item.ordinal),
                        ],
                        "rows": item.rows,
                        "bytes": item.byte_count,
                    }
                    for item in snapshot.outputs
                ],
                "modelRef": result.get("modelRef"),
                "checkpoint": result.get("checkpoint"),
            },
            pollAfterMs=0 if terminal else 500,
        )


def _safe_recovery(recovery) -> dict | None:
    if recovery is None:
        return None
    checkpoint = recovery.checkpoint
    return {
        "latestCheckpoint": (
            None
            if checkpoint is None
            else {
                "generation": checkpoint.generation,
                "completedEpochs": checkpoint.completed_epochs,
                "globalStep": checkpoint.global_step,
                "trainingComplete": checkpoint.training_complete,
            }
        ),
        "resumedFromGeneration": recovery.resumed_from_generation,
        "retryCount": recovery.retry_count,
        "lastRetryCode": recovery.last_retry_code,
        "boundary": "globalEpoch",
    }


def _safe_input(item) -> dict:
    return {
        "payloadId": item.payload_id,
        "ordinal": item.ordinal,
        "schemaId": item.schema_id,
        "rows": item.rows,
        "batches": item.batches,
        "bytes": item.byte_count,
        "sha256": item.sha256,
        "schemaFingerprint": item.schema_fingerprint,
    }


def _timestamps(job) -> dict:
    return {
        "createdAt": _timestamp(job.created_at),
        "updatedAt": _timestamp(job.updated_at),
        "sealedAt": _timestamp(job.sealed_at),
        "queuedAt": _timestamp(job.queued_at),
        "startedAt": _timestamp(job.started_at),
        "cancelRequestedAt": _timestamp(job.cancel_requested_at),
        "finishedAt": _timestamp(job.finished_at),
    }


def _timestamp(value) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = ["GetJobStatus"]
