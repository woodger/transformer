from __future__ import annotations

from datetime import UTC, datetime

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v3 import ModelContract
from app.service.adapters.inbound.flight.constants import (
    CONTRACT_PATH_VERSION,
)
from app.service.adapters.inbound.flight.devices import device_to_api
from app.service.adapters.inbound.flight.documents import (
    response_document,
)
from app.service.adapters.inbound.flight.model_catalog import (
    model_detail,
    model_summary,
)
from app.service.adapters.inbound.flight.mutation_lease import (
    encode_mutation_lease,
)
from app.service.application.messages.jobs import (
    InputClosed,
    JobAcquired,
    JobCancelled,
    JobCreated,
    JobInputsPage,
    JobOutputsPage,
    JobStatusResult,
    ServiceLimits,
)
from app.service.application.messages.model_catalog import (
    CatalogModelDetail,
    CatalogModelsPage,
)
from app.service.domain.job import TERMINAL_EXECUTION_STATES, ExecutionState
from app.service.domain.records import (
    JobRecord,
)


def present_job_created(result: JobCreated) -> JsonObject:
    common = {
        "jobId": result.job_id,
        "mutationLease": encode_mutation_lease(
            result.client_execution_id,
            result.fencing_token,
        ),
        "input": {
            "state": result.input_state.value,
            "revision": result.input_revision,
            "nextOrdinal": result.next_input_ordinal,
        },
        "execution": {"state": result.execution_state.value},
        "device": {
            "requested": device_to_api(result.requested_device),
            "selected": device_to_api(result.selected_device),
        },
        "upload": {
            "descriptorPath": [
                "transformer",
                CONTRACT_PATH_VERSION,
                "jobs",
                result.job_id,
                "inputs",
                "{ordinal}",
            ],
            "oneDoPutIsOnePhysicalPayload": True,
        },
        "limits": limits_to_api(result.limits),
    }
    if result.operation == "fit":
        return response_document(
            result.request_id,
            **common,
            resolvedDefinition={
                "dataBinding": _data_binding_to_api(
                    result.data_contract,
                    result.source_encoding,
                ),
                "modelContract": dict(result.model_contract),
                "semanticDigests": dict(result.semantic_digests),
                "requestedInitialization": _requested_initialization(
                    result.initialization,
                ),
            },
        )
    if result.resolved_model_ref is None:
        raise ValueError("predict job is missing its resolved model reference")
    model_contract = ModelContract.from_document(result.model_contract)
    targets: list[JsonObject] = []
    for slot in model_contract.target_slots:
        identity = slot["identity"]
        transformation = slot["publicPredictionTransformation"]
        if not isinstance(identity, str) or not isinstance(transformation, str):
            raise ValueError("model target slots are invalid")
        targets.append(
            {
                "identity": identity,
                "publicPredictionTransformation": transformation,
            },
        )
    return response_document(
        result.request_id,
        **common,
        modelRef=result.resolved_model_ref,
        predictionDefinition={
            "seqLen": result.data_contract["seqLen"],
            "outputWidth": model_contract.target_width,
            "targets": targets,
        },
    )


def present_job_acquired(result: JobAcquired) -> JsonObject:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        revision=result.revision,
        mutationLease=encode_mutation_lease(
            result.client_execution_id,
            result.fencing_token,
        ),
    )


def present_input_closed(result: InputClosed) -> JsonObject:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        revision=result.revision,
        input={
            "state": result.input_state.value,
            "payloadCount": result.payload_count,
            "logicalRows": result.total_rows,
            "manifestSha256": result.manifest_sha256,
        },
        execution={"state": result.execution_state.value},
    )


def present_job_cancelled(result: JobCancelled) -> JsonObject:
    return response_document(
        result.request_id,
        jobId=result.job_id,
        revision=result.revision,
        input={"state": result.input_state.value},
        execution={"state": result.execution_state.value},
    )


def present_job_status(result: JobStatusResult) -> JsonObject:
    snapshot = result.snapshot
    job = snapshot.job
    if job is None:
        raise ValueError("job status result is missing its job")
    terminal = job.execution_state in TERMINAL_EXECUTION_STATES
    durable_result = job.result or {}
    error: JsonObject | None = None
    if job.execution_state == ExecutionState.FAILED:
        error = _error_to_api(job.error_code, job.error_message)
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
            "logicalRows": job.total_rows,
            "manifestSha256": job.manifest_sha256,
        },
        execution={
            "state": job.execution_state.value,
            "attempt": job.attempt,
        },
        timestamps=_timestamps(job),
        device={
            "requested": device_to_api(job.requested_device),
            "selected": device_to_api(job.selected_device),
        },
        progress=job.progress,
        error=error,
        results={
            "outputCount": snapshot.output_count,
            "modelRef": durable_result.get("modelRef"),
        },
        pollAfterMs=0 if terminal else 500,
    )


def present_job_inputs(result: JobInputsPage) -> JsonObject:
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
                "chunks": item.chunks,
                "logicalRows": item.rows,
                "nativeRows": list(item.native_rows),
                "firstRangeOrdinal": item.first_range_ordinal,
                "firstExampleOffset": item.first_example_offset,
                "lastRangeOrdinal": item.last_range_ordinal,
                "nextExampleOffset": item.next_example_offset,
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


def present_job_outputs(result: JobOutputsPage) -> JsonObject:
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
                "logicalRows": item.rows,
                "bytes": item.byte_count,
                "sha256": item.sha256,
            }
            for item in result.items
        ],
        nextCursor=result.next_cursor,
        hasMore=result.has_more,
    )


def present_catalog_models_page(result: CatalogModelsPage) -> JsonObject:
    return {
        "requestId": result.request_id,
        "models": [model_summary(model) for model in result.models],
        "nextCursor": result.next_cursor,
        "cursorExpiresAt": _catalog_timestamp(result.cursor_expires_at),
    }


def present_catalog_model_detail(result: CatalogModelDetail) -> JsonObject:
    return {
        "requestId": result.request_id,
        "model": model_detail(result.model),
    }


def limits_to_api(limits: ServiceLimits) -> JsonObject:
    return {
        "maxPayloadBytes": limits.max_payload_bytes,
        "maxRowsPerPayload": limits.max_rows_per_payload,
        "maxPayloadsPerJob": limits.max_payloads_per_job,
        "maxJobBytes": limits.max_job_bytes,
        "inputIdleTimeoutSeconds": limits.input_idle_timeout_seconds,
    }


def _data_binding_to_api(
    data_contract: JsonObject,
    input_layout: JsonObject,
) -> JsonObject:
    return {
        "dataContractSha256": data_contract["dataContractSha256"],
        "tensorGeometry": {
            "seqLen": data_contract["seqLen"],
            "featureDim": data_contract["featureDim"],
        },
        "inputLayout": dict(input_layout),
    }


def _requested_initialization(
    initialization: JsonObject | None,
) -> JsonObject:
    if initialization is None or initialization.get("source") == "random":
        return {"source": "random"}
    parent_model_ref = initialization.get("parentModelRef")
    if not isinstance(parent_model_ref, str):
        raise ValueError("published-model initialization is invalid")
    return {
        "source": "publishedModel",
        "modelRef": parent_model_ref,
    }


def _error_to_api(code: str | None, message: str | None) -> JsonObject:
    if code in ("CUDA_OUT_OF_MEMORY", "GPU_OUT_OF_MEMORY"):
        return {
            "code": "GPU_OUT_OF_MEMORY",
            "message": "GPU execution ran out of memory",
            "detail": None,
        }
    if message is not None and "cuda" in message.lower():
        if code == "DEVICE_LOST":
            message = "GPU device became unavailable during execution"
        elif code == "SUBPROCESS_FAILED":
            message = "GPU subprocess failed"
        else:
            message = "GPU execution failed"
    return {"code": code, "message": message, "detail": None}


def _timestamps(job: JobRecord) -> JsonObject:
    return {
        "createdAt": _timestamp(job.created_at),
        "updatedAt": _timestamp(job.updated_at),
        "inputClosedAt": _timestamp(job.input_closed_at),
        "queuedAt": _timestamp(job.queued_at),
        "startedAt": _timestamp(job.started_at),
        "cancelRequestedAt": _timestamp(job.cancel_requested_at),
        "finishedAt": _timestamp(job.finished_at),
    }


def _timestamp(value: float | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value, tz=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _catalog_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "limits_to_api",
    "present_catalog_model_detail",
    "present_catalog_models_page",
    "present_input_closed",
    "present_job_acquired",
    "present_job_cancelled",
    "present_job_created",
    "present_job_inputs",
    "present_job_outputs",
    "present_job_status",
]
