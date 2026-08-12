from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.service.application.job_models import (
    AcquireJobCommand,
    CancelJobCommand,
    CloseInputCommand,
    CreateJobCommand,
    InputClosed,
    JobAcquired,
    JobCancelled,
    JobCreated,
    JobCreationPreparation,
    ServiceLimits,
)
from app.service.application.ports.job_lifecycle import (
    ArtifactLocation,
    LifecycleMutation,
)
from app.service.domain.errors import ServiceError, conflict, not_found
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import PublishedModelRecord


@dataclass(frozen=True, slots=True)
class JobActionNames:
    create: str
    acquire: str
    input_close: str
    cancel: str


class PostgresJobLifecycle:
    """Keep idempotency and job mutation transactions inside PostgreSQL."""

    def __init__(self, ledger, action_names: JobActionNames):
        self.ledger = ledger
        self.action_names = action_names

    def create(
        self,
        command: CreateJobCommand,
        *,
        max_active_jobs: int,
        preflight: Callable[[], None],
        prepare: Callable[
            [PublishedModelRecord | None],
            JobCreationPreparation,
        ],
    ) -> LifecycleMutation[JobCreated]:
        replay = self.ledger.lookup_idempotency(
            command.owner_subject,
            self.action_names.create,
            command.idempotency_key,
        )
        if replay is not None:
            _require_request_hash(replay, command.request_hash)
            return LifecycleMutation(
                _decode_created(replay["response"]),
                replayed=True,
            )

        identity = self.ledger.get_job_identity(command.job_id)
        if identity is not None:
            _require_matching_identity(identity, command)
            return LifecycleMutation(
                _decode_created(identity["create_result"]),
                replayed=True,
            )

        preflight()
        created = False

        def mutation(connection):
            nonlocal created
            self.ledger.lock_job_identity(
                command.job_id,
                connection=connection,
            )
            durable_identity = self.ledger.get_job_identity(
                command.job_id,
                connection=connection,
            )
            if durable_identity is not None:
                _require_matching_identity(durable_identity, command)
                return durable_identity["create_result"], command.job_id

            active = self.ledger.active_job_count(
                command.owner_subject,
                connection=connection,
            )
            if active >= max_active_jobs:
                raise ServiceError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "active job quota exceeded",
                )

            model = None
            if command.operation == "predict":
                model = self._resolve_model(command, connection)
            prepared = prepare(model)
            encoded = _encode_created(prepared.result)
            self.ledger.create_job(
                job_id=command.job_id,
                owner_subject=command.owner_subject,
                client_execution_id=command.client_execution_id,
                operation=command.operation,
                requested_device=command.requested_device,
                prediction_column=command.prediction_column,
                config_hash=command.request_hash,
                data_contract=command.data_contract,
                ml_contract=command.ml_contract,
                create_result=encoded,
                model_label=command.model_label,
                resolved_model_ref=prepared.resolved_model_ref,
                model_config=prepared.model_config,
                training_config=prepared.training_config,
                connection=connection,
            )
            created = True
            return encoded, command.job_id

        document, replayed = self.ledger.run_idempotent(
            owner_subject=command.owner_subject,
            action_name=self.action_names.create,
            idempotency_key=command.idempotency_key,
            request_hash=command.request_hash,
            mutation=mutation,
        )
        return LifecycleMutation(
            _decode_created(document),
            replayed=replayed or not created,
        )

    def acquire(
        self,
        command: AcquireJobCommand,
        *,
        acquire_grace_seconds: float,
    ) -> LifecycleMutation[JobAcquired]:
        cleanup: tuple[tuple[str, str], ...] = ()

        def mutation(connection):
            nonlocal cleanup
            job, cleanup = self.ledger.acquire_job(
                command.job_id,
                owner_subject=command.owner_subject,
                previous_client_execution_id=(
                    command.previous_client_execution_id
                ),
                expected_fencing_token=command.expected_fencing_token,
                client_execution_id=command.client_execution_id,
                acquire_grace_seconds=acquire_grace_seconds,
                connection=connection,
            )
            result = JobAcquired(
                request_id=command.request_id,
                job_id=job["job_id"],
                revision=job["revision"],
                client_execution_id=job["client_execution_id"],
                fencing_token=job["fencing_token"],
            )
            return _encode_acquired(result), job["job_id"]

        document, replayed = self.ledger.run_idempotent(
            owner_subject=command.owner_subject,
            action_name=self.action_names.acquire,
            idempotency_key=command.idempotency_key,
            request_hash=command.request_hash,
            mutation=mutation,
        )
        return LifecycleMutation(
            _decode_acquired(document),
            replayed=replayed,
            cleanup=() if replayed else _locations(cleanup),
        )

    def close_input(
        self,
        command: CloseInputCommand,
        *,
        select_device: Callable[[str, str | None, str, int], str],
    ) -> LifecycleMutation[InputClosed]:
        queued = False

        def mutation(connection):
            nonlocal queued
            current = self.ledger.get_job(
                command.job_id,
                owner_subject=command.owner_subject,
                connection=connection,
                for_update=True,
            )
            if current is None:
                raise not_found("job not found")
            selected = select_device(
                current["requested_device"],
                current["selected_device"],
                current["operation"],
                command.total_rows,
            )
            job, repeated = self.ledger.close_input(
                command.job_id,
                client_execution_id=command.client_execution_id,
                fencing_token=command.fencing_token,
                payload_count=command.payload_count,
                total_rows=command.total_rows,
                total_bytes=command.total_bytes,
                manifest_sha256=command.manifest_sha256,
                selected_device=selected,
                connection=connection,
            )
            queued = (
                not repeated
                and current["execution_state"]
                == ExecutionState.WAITING_INPUT.value
                and job["execution_state"] == ExecutionState.QUEUED.value
            )
            result = InputClosed(
                request_id=command.request_id,
                job_id=job["job_id"],
                revision=job["revision"],
                input_state=InputState(job["input_state"]),
                input_revision=job["input_revision"],
                payload_count=job["payload_count"],
                total_rows=job["total_rows"],
                total_bytes=job["total_bytes"],
                manifest_sha256=job["manifest_sha256"],
                execution_state=ExecutionState(job["execution_state"]),
            )
            return _encode_closed(result), job["job_id"]

        document, replayed = self.ledger.run_idempotent(
            owner_subject=command.owner_subject,
            action_name=self.action_names.input_close,
            idempotency_key=command.idempotency_key,
            request_hash=command.request_hash,
            mutation=mutation,
            replay_guard=lambda connection: _require_current_fence(
                self.ledger,
                command,
                connection,
            ),
        )
        return LifecycleMutation(
            _decode_closed(document),
            replayed=replayed,
            queued=not replayed and queued,
        )

    def cancel(
        self,
        command: CancelJobCommand,
    ) -> LifecycleMutation[JobCancelled]:
        notify = False
        cleanup: tuple[tuple[str, str], ...] = ()

        def mutation(connection):
            nonlocal notify, cleanup
            job, notify, cleanup = self.ledger.cancel_job(
                command.job_id,
                owner_subject=command.owner_subject,
                client_execution_id=command.client_execution_id,
                fencing_token=command.fencing_token,
                connection=connection,
            )
            result = JobCancelled(
                request_id=command.request_id,
                job_id=job["job_id"],
                revision=job["revision"],
                input_state=InputState(job["input_state"]),
                execution_state=ExecutionState(job["execution_state"]),
            )
            return _encode_cancelled(result), job["job_id"]

        document, replayed = self.ledger.run_idempotent(
            owner_subject=command.owner_subject,
            action_name=self.action_names.cancel,
            idempotency_key=command.idempotency_key,
            request_hash=command.request_hash,
            mutation=mutation,
            replay_guard=lambda connection: _require_current_fence(
                self.ledger,
                command,
                connection,
            ),
        )
        return LifecycleMutation(
            _decode_cancelled(document),
            replayed=replayed,
            cleanup=() if replayed else _locations(cleanup),
            notify_worker=not replayed and notify,
        )

    def _resolve_model(self, command, connection):
        if command.model_selector == "alias":
            model = self.ledger.resolve_published_model_alias(
                command.owner_subject,
                command.model_ref,
                connection=connection,
            )
        else:
            model = self.ledger.get_published_model(
                command.model_ref,
                owner_subject=command.owner_subject,
                connection=connection,
            )
        if model is None:
            raise not_found("model generation not found")
        return model

def _require_request_hash(record: dict, request_hash: str) -> None:
    if record["request_hash"] != request_hash:
        raise conflict("idempotency key was used for a different request")


def _require_matching_identity(identity: dict, command) -> None:
    if (
        identity["owner_subject"] != command.owner_subject
        or identity["create_hash"] != command.request_hash
    ):
        raise conflict("jobId has already been used for a different request")


def _require_current_fence(ledger, command, connection) -> None:
    job = ledger.get_job(
        command.job_id,
        owner_subject=command.owner_subject,
        connection=connection,
        for_update=True,
    )
    if job is None:
        raise not_found("job not found")
    if (
        job["client_execution_id"] != command.client_execution_id
        or job["fencing_token"] != command.fencing_token
    ):
        raise ServiceError(
            ErrorCode.STALE_FENCE,
            "job ownership fence is stale",
        )


def _locations(values) -> tuple[ArtifactLocation, ...]:
    return tuple(ArtifactLocation(*value) for value in values)


def _encode_created(result: JobCreated) -> dict:
    return {
        "result_type": "job_created",
        "request_id": result.request_id,
        "job_id": result.job_id,
        "operation": result.operation,
        "revision": result.revision,
        "input_state": result.input_state.value,
        "input_revision": result.input_revision,
        "next_input_ordinal": result.next_input_ordinal,
        "execution_state": result.execution_state.value,
        "client_execution_id": result.client_execution_id,
        "fencing_token": result.fencing_token,
        "requested_device": result.requested_device,
        "selected_device": result.selected_device,
        "resolved_model_ref": result.resolved_model_ref,
        "data_contract": dict(result.data_contract),
        "ml_contract": dict(result.ml_contract),
        "limits": _encode_limits(result.limits),
    }


def _decode_created(document: dict) -> JobCreated:
    if document.get("result_type") == "job_created":
        return JobCreated(
            request_id=document["request_id"],
            job_id=document["job_id"],
            operation=document["operation"],
            revision=document["revision"],
            input_state=InputState(document["input_state"]),
            input_revision=document["input_revision"],
            next_input_ordinal=document["next_input_ordinal"],
            execution_state=ExecutionState(document["execution_state"]),
            client_execution_id=document["client_execution_id"],
            fencing_token=document["fencing_token"],
            requested_device=document["requested_device"],
            selected_device=document["selected_device"],
            resolved_model_ref=document.get("resolved_model_ref"),
            data_contract=dict(document["data_contract"]),
            ml_contract=dict(document["ml_contract"]),
            limits=_decode_limits(document["limits"]),
        )
    ownership = document["ownership"]
    return JobCreated(
        request_id=document["requestId"],
        job_id=document["jobId"],
        operation=document["operation"],
        revision=document["revision"],
        input_state=InputState(document["input"]["state"]),
        input_revision=document["input"]["revision"],
        next_input_ordinal=document["input"]["nextOrdinal"],
        execution_state=ExecutionState(document["execution"]["state"]),
        client_execution_id=ownership["clientExecutionId"],
        fencing_token=int(ownership["fencingToken"]),
        requested_device=document["device"]["requested"],
        selected_device=document["device"].get("selected"),
        resolved_model_ref=document.get("resolvedModelRef"),
        data_contract=_wire_data_contract(document["dataContract"]),
        ml_contract=dict(document["mlContract"]),
        limits=_wire_limits(document["limits"]),
    )


def _encode_acquired(result: JobAcquired) -> dict:
    return {
        "result_type": "job_acquired",
        "request_id": result.request_id,
        "job_id": result.job_id,
        "revision": result.revision,
        "client_execution_id": result.client_execution_id,
        "fencing_token": result.fencing_token,
    }


def _decode_acquired(document: dict) -> JobAcquired:
    if document.get("result_type") == "job_acquired":
        return JobAcquired(
            request_id=document["request_id"],
            job_id=document["job_id"],
            revision=document["revision"],
            client_execution_id=document["client_execution_id"],
            fencing_token=document["fencing_token"],
        )
    return JobAcquired(
        request_id=document["requestId"],
        job_id=document["jobId"],
        revision=document["revision"],
        client_execution_id=document["ownership"]["clientExecutionId"],
        fencing_token=int(document["ownership"]["fencingToken"]),
    )


def _encode_closed(result: InputClosed) -> dict:
    return {
        "result_type": "input_closed",
        "request_id": result.request_id,
        "job_id": result.job_id,
        "revision": result.revision,
        "input_state": result.input_state.value,
        "input_revision": result.input_revision,
        "payload_count": result.payload_count,
        "total_rows": result.total_rows,
        "total_bytes": result.total_bytes,
        "manifest_sha256": result.manifest_sha256,
        "execution_state": result.execution_state.value,
    }


def _decode_closed(document: dict) -> InputClosed:
    if document.get("result_type") == "input_closed":
        return InputClosed(
            request_id=document["request_id"],
            job_id=document["job_id"],
            revision=document["revision"],
            input_state=InputState(document["input_state"]),
            input_revision=document["input_revision"],
            payload_count=document["payload_count"],
            total_rows=document["total_rows"],
            total_bytes=document["total_bytes"],
            manifest_sha256=document["manifest_sha256"],
            execution_state=ExecutionState(document["execution_state"]),
        )
    input_document = document["input"]
    return InputClosed(
        request_id=document["requestId"],
        job_id=document["jobId"],
        revision=document["revision"],
        input_state=InputState(input_document["state"]),
        input_revision=input_document["revision"],
        payload_count=input_document["payloadCount"],
        total_rows=input_document["totalRows"],
        total_bytes=input_document["totalBytes"],
        manifest_sha256=input_document["manifestSha256"],
        execution_state=ExecutionState(document["execution"]["state"]),
    )


def _encode_cancelled(result: JobCancelled) -> dict:
    return {
        "result_type": "job_cancelled",
        "request_id": result.request_id,
        "job_id": result.job_id,
        "revision": result.revision,
        "input_state": result.input_state.value,
        "execution_state": result.execution_state.value,
    }


def _decode_cancelled(document: dict) -> JobCancelled:
    if document.get("result_type") == "job_cancelled":
        return JobCancelled(
            request_id=document["request_id"],
            job_id=document["job_id"],
            revision=document["revision"],
            input_state=InputState(document["input_state"]),
            execution_state=ExecutionState(document["execution_state"]),
        )
    return JobCancelled(
        request_id=document["requestId"],
        job_id=document["jobId"],
        revision=document["revision"],
        input_state=InputState(document["input"]["state"]),
        execution_state=ExecutionState(document["execution"]["state"]),
    )


def _encode_limits(limits: ServiceLimits) -> dict:
    return {
        name: getattr(limits, name)
        for name in ServiceLimits.__dataclass_fields__
    }


def _decode_limits(document: dict) -> ServiceLimits:
    return ServiceLimits(**{
        name: document[name]
        for name in ServiceLimits.__dataclass_fields__
    })


def _wire_limits(document: dict) -> ServiceLimits:
    return ServiceLimits(
        max_message_bytes=document["maxMessageBytes"],
        target_batch_bytes=document["targetBatchBytes"],
        max_batch_bytes=document["maxBatchBytes"],
        max_payload_bytes=document["maxPayloadBytes"],
        max_rows_per_payload=document["maxRowsPerPayload"],
        max_payloads_per_job=document["maxPayloadsPerJob"],
        max_job_bytes=document["maxJobBytes"],
        max_active_jobs_per_subject=document["maxActiveJobsPerSubject"],
        max_page_items=document["maxPageItems"],
        input_idle_timeout_seconds=document["inputIdleTimeoutSeconds"],
    )


def _wire_data_contract(document: dict) -> dict:
    return {
        "id": document["id"],
        "version": document["version"],
        "data_contract_sha256": document["dataContractSha256"],
        "seq_len": document["seqLen"],
        "feature_dim": document["featureDim"],
        "target_schema_id": document["targetSchemaId"],
    }


__all__ = ["JobActionNames", "PostgresJobLifecycle"]
