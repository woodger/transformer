from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.service.domain.errors import ServiceError, not_found
from app.service.domain.job import TERMINAL_EXECUTION_STATES, ErrorCode, ExecutionState


class GetJobStatus:
    """Read one bounded, consistent durable status snapshot."""

    def __init__(
        self,
        ledger,
        *,
        response_factory: Callable[..., dict],
        data_contract_factory: Callable[[dict], dict],
    ):
        self.ledger = ledger
        self._response = response_factory
        self._data_contract = data_contract_factory

    def execute(self, owner: str, job_id: str, request_id: str) -> dict:
        snapshot = self.ledger.get_status_snapshot_record(job_id, owner)
        job = snapshot.job
        if job is None:
            identity = self.ledger.get_job_identity(
                job_id,
                owner_subject=owner,
            )
            if identity is not None and identity["retired_at"] is not None:
                raise ServiceError(
                    ErrorCode.JOB_RETIRED,
                    "job identity has been retired",
                )
            raise not_found("job not found")
        terminal = job.execution_state in TERMINAL_EXECUTION_STATES
        result = job.result or {}
        error = None
        if job.execution_state == ExecutionState.FAILED:
            error = {"code": job.error_code, "message": job.error_message}
        return self._response(
            request_id,
            jobId=job_id,
            operation=job.operation,
            revision=job.revision,
            input={
                "state": job.input_state.value,
                "revision": job.input_revision,
                "nextOrdinal": job.next_input_ordinal,
                "payloadCount": job.payload_count,
                "totalRows": job.total_rows,
                "totalBytes": job.total_bytes,
                "manifestSha256": job.manifest_sha256,
            },
            execution={
                "state": job.execution_state.value,
                "attempt": job.attempt,
            },
            ownership={
                "clientExecutionId": job.client_execution_id,
                "fencingToken": str(job.fencing_token),
            },
            timestamps=_timestamps(job),
            device={
                "requested": job.requested_device,
                "selected": job.selected_device,
            },
            dataContract=self._data_contract(job.data_contract),
            resolvedModelRef=job.resolved_model_ref,
            predictionColumn=job.prediction_column,
            progress=job.progress,
            recovery=_safe_recovery(snapshot.recovery),
            error=error,
            results={
                "outputCount": snapshot.output_count,
                "modelRef": result.get("modelRef"),
                "checkpoint": result.get("checkpoint"),
            },
            pollAfterMs=0 if terminal else 500,
        )


class ListJobInputs:
    def __init__(self, ledger, *, response_factory: Callable[..., dict]):
        self.ledger = ledger
        self._response = response_factory

    def execute(self, owner: str, request: dict) -> dict:
        page = self.ledger.list_inputs_page(
            request["job_id"],
            owner,
            after_revision=request["after_revision"],
            snapshot_revision=request["snapshot_revision"],
            cursor=request["cursor"],
            limit=request["limit"],
        )
        return self._response(
            request["request_id"],
            jobId=request["job_id"],
            afterRevision=page["after_revision"],
            snapshotRevision=page["snapshot_revision"],
            cursor=page["cursor"],
            items=[_input_receipt(item) for item in page["items"]],
            nextCursor=page["next_cursor"],
            hasMore=page["has_more"],
        )


class ListJobOutputs:
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

    def execute(self, owner: str, request: dict) -> dict:
        page = self.ledger.list_outputs_page(
            request["job_id"],
            owner,
            cursor=request["cursor"],
            limit=request["limit"],
        )
        return self._response(
            request["request_id"],
            jobId=request["job_id"],
            cursor=page["cursor"],
            items=[
                {
                    "ordinal": item["ordinal"],
                    "descriptorPath": [
                        "transformer",
                        self.path_version,
                        "jobs",
                        request["job_id"],
                        "outputs",
                        str(item["ordinal"]),
                    ],
                    "rows": item["rows"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for item in page["items"]
            ],
            nextCursor=page["next_cursor"],
            hasMore=page["has_more"],
        )


class DescribeModel:
    def __init__(
        self,
        ledger,
        *,
        response_factory: Callable[..., dict],
        data_contract_factory: Callable[[dict], dict],
        model_config_factory: Callable[[dict], dict],
        model_artifact_validator: Callable[[object], None],
    ):
        self.ledger = ledger
        self._response = response_factory
        self._data_contract = data_contract_factory
        self._model_config = model_config_factory
        self._validate_model_artifact = model_artifact_validator

    def execute(self, owner: str, request: dict) -> dict:
        if request["model_selector"] == "modelAlias":
            model = self.ledger.resolve_published_model_alias(
                owner,
                request["model_ref"],
            )
        else:
            model = self.ledger.get_published_model(
                request["model_ref"],
                owner_subject=owner,
            )
        if model is None:
            raise not_found("model generation not found")
        if not model.certified_for_v3 or model.data_contract is None:
            raise ServiceError(
                ErrorCode.MODEL_SCHEMA_MISMATCH,
                "model generation is not certified for Flight v3",
            )
        self._validate_model_artifact(model)
        try:
            model_config = self._model_config(model.metadata["model_config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model metadata is invalid",
            ) from exc
        return self._response(
            request["request_id"],
            modelRef=model.model_ref,
            label=model.label,
            generation=model.generation,
            dataContract=self._data_contract(model.data_contract),
            modelConfig=model_config,
            checkpoint={
                "format": "transformer-checkpoint-v2",
                "sha256": model.sha256,
                "bytes": model.byte_count,
            },
            createdAt=_timestamp(model.created_at),
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


def _input_receipt(item: dict) -> dict:
    return {
        "payloadId": item["payload_id"],
        "ordinal": item["ordinal"],
        "commitRevision": item["commit_revision"],
        "schemaId": item["schema_id"],
        "dataContractSha256": item["data_contract_sha256"],
        "rows": item["rows"],
        "batches": item["batches"],
        "bytes": item["bytes"],
        "sha256": item["sha256"],
        "schemaFingerprint": item["schema_fingerprint"],
    }


def _timestamps(job) -> dict:
    return {
        "createdAt": _timestamp(job.created_at),
        "updatedAt": _timestamp(job.updated_at),
        "inputClosedAt": _timestamp(job.input_closed_at),
        "queuedAt": _timestamp(job.queued_at),
        "startedAt": _timestamp(job.started_at),
        "cancelRequestedAt": _timestamp(job.cancel_requested_at),
        "finishedAt": _timestamp(job.finished_at),
    }


def _timestamp(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value, tz=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "DescribeModel",
    "GetJobStatus",
    "ListJobInputs",
    "ListJobOutputs",
]
