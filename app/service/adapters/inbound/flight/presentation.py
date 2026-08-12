from __future__ import annotations

from datetime import UTC, datetime

from app.contracts.worker.v3.objective import CHECKPOINT_FORMAT
from app.service.adapters.inbound.flight.constants import (
    CONTRACT_PATH_VERSION,
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
)
from app.service.adapters.inbound.flight.contract import (
    data_contract_to_api,
    model_config_to_api,
    response_document,
)
from app.service.application.job_models import (
    InputClosed,
    JobAcquired,
    JobCancelled,
    JobCreated,
    JobInputsPage,
    JobOutputsPage,
    JobStatusResult,
    ModelDescription,
    ServiceLimits,
)
from app.service.domain.job import TERMINAL_EXECUTION_STATES, ExecutionState


def present_job_created(result: JobCreated) -> dict:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        operation=result.operation,
        revision=result.revision,
        input={
            "state": result.input_state.value,
            "revision": result.input_revision,
            "nextOrdinal": result.next_input_ordinal,
        },
        execution={"state": result.execution_state.value},
        ownership={
            "clientExecutionId": result.client_execution_id,
            "fencingToken": str(result.fencing_token),
        },
        device={
            "requested": result.requested_device,
            "selected": result.selected_device,
        },
        resolvedModelRef=result.resolved_model_ref,
        dataContract=data_contract_to_api(result.data_contract),
        mlContract=dict(result.ml_contract),
        limits=limits_to_api(result.limits),
        upload={
            "descriptorPath": [
                "transformer",
                CONTRACT_PATH_VERSION,
                "jobs",
                result.job_id,
                "inputs",
                "{ordinal}",
            ],
            "schemaId": (
                FIT_SCHEMA_ID
                if result.operation == "fit"
                else PREDICT_SCHEMA_ID
            ),
            "oneDoPutIsOneSemanticPayload": True,
        },
    )


def present_job_acquired(result: JobAcquired) -> dict:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        revision=result.revision,
        ownership={
            "clientExecutionId": result.client_execution_id,
            "fencingToken": str(result.fencing_token),
        },
    )


def present_input_closed(result: InputClosed) -> dict:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        revision=result.revision,
        input={
            "state": result.input_state.value,
            "revision": result.input_revision,
            "payloadCount": result.payload_count,
            "totalRows": result.total_rows,
            "totalBytes": result.total_bytes,
            "manifestSha256": result.manifest_sha256,
        },
        execution={"state": result.execution_state.value},
    )


def present_job_cancelled(result: JobCancelled) -> dict:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        revision=result.revision,
        input={"state": result.input_state.value},
        execution={"state": result.execution_state.value},
    )


def present_job_status(result: JobStatusResult) -> dict:
    snapshot = result.snapshot
    job = snapshot.job
    if job is None:
        raise ValueError("job status result is missing its job")
    terminal = job.execution_state in TERMINAL_EXECUTION_STATES
    durable_result = job.result or {}
    error = None
    if job.execution_state == ExecutionState.FAILED:
        error = {"code": job.error_code, "message": job.error_message}
    return response_document(
        result.request_id,
        jobId=job.job_id,
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
        dataContract=data_contract_to_api(job.data_contract),
        mlContract=dict(job.ml_contract),
        resolvedModelRef=job.resolved_model_ref,
        predictionColumn=job.prediction_column,
        progress=job.progress,
        recovery=_safe_recovery(snapshot.recovery),
        error=error,
        results={
            "outputCount": snapshot.output_count,
            "modelRef": durable_result.get("modelRef"),
            "checkpoint": durable_result.get("checkpoint"),
        },
        pollAfterMs=0 if terminal else 500,
    )


def present_job_inputs(result: JobInputsPage) -> dict:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        afterRevision=result.after_revision,
        snapshotRevision=result.snapshot_revision,
        cursor=result.cursor,
        items=[
            {
                "payloadId": item.payload_id,
                "ordinal": item.ordinal,
                "commitRevision": item.commit_revision,
                "schemaId": item.schema_id,
                "dataContractSha256": item.data_contract_sha256,
                "rows": item.rows,
                "batches": item.batches,
                "bytes": item.byte_count,
                "sha256": item.sha256,
                "schemaFingerprint": item.schema_fingerprint,
            }
            for item in result.items
        ],
        nextCursor=result.next_cursor,
        hasMore=result.has_more,
    )


def present_job_outputs(result: JobOutputsPage) -> dict:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        cursor=result.cursor,
        items=[
            {
                "ordinal": item.ordinal,
                "descriptorPath": [
                    "transformer",
                    CONTRACT_PATH_VERSION,
                    "jobs",
                    result.job_id,
                    "outputs",
                    str(item.ordinal),
                ],
                "rows": item.rows,
                "bytes": item.byte_count,
                "sha256": item.sha256,
            }
            for item in result.items
        ],
        nextCursor=result.next_cursor,
        hasMore=result.has_more,
    )


def present_model_description(result: ModelDescription) -> dict:
    model = result.model
    return response_document(
        result.request_id,
        modelRef=model.model_ref,
        label=model.label,
        generation=model.generation,
        dataContract=data_contract_to_api(model.data_contract),
        mlContract=dict(model.ml_contract),
        modelConfig=model_config_to_api(result.model_config),
        checkpoint={
            "format": CHECKPOINT_FORMAT,
            "sha256": model.sha256,
            "bytes": model.byte_count,
        },
        createdAt=_timestamp(model.created_at),
    )


def limits_to_api(limits: ServiceLimits) -> dict:
    return {
        "maxMessageBytes": limits.max_message_bytes,
        "targetBatchBytes": limits.target_batch_bytes,
        "maxBatchBytes": limits.max_batch_bytes,
        "maxPayloadBytes": limits.max_payload_bytes,
        "maxRowsPerPayload": limits.max_rows_per_payload,
        "maxPayloadsPerJob": limits.max_payloads_per_job,
        "maxJobBytes": limits.max_job_bytes,
        "maxActiveJobsPerSubject": limits.max_active_jobs_per_subject,
        "maxPageItems": limits.max_page_items,
        "inputIdleTimeoutSeconds": limits.input_idle_timeout_seconds,
        "transportMessageLimitEnforced": False,
    }


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
    "limits_to_api",
    "present_input_closed",
    "present_job_acquired",
    "present_job_cancelled",
    "present_job_created",
    "present_job_inputs",
    "present_job_outputs",
    "present_job_status",
    "present_model_description",
]
