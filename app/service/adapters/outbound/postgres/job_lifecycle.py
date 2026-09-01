from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.contracts.json_types import JsonObject, JsonValue
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.mapping import (
    row_integer,
    row_json_object,
    row_optional_string,
    row_string,
)
from app.service.application.messages.jobs import (
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

    def __init__(self, ledger: Ledger, action_names: JobActionNames) -> None:
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
                _decode_created(row_json_object(replay, "response")),
                replayed=True,
            )

        identity = self.ledger.get_job_identity(command.job_id)
        if identity is not None:
            _require_matching_identity(identity, command)
            return LifecycleMutation(
                _decode_created(row_json_object(identity, "create_result")),
                replayed=True,
            )

        preflight()
        created = False

        def mutation(connection: Session) -> tuple[JsonObject, str | None]:
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
                return (
                    row_json_object(durable_identity, "create_result"),
                    command.job_id,
                )

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
            if command.model_ref is not None:
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
                initialization=prepared.result.initialization,
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

        def mutation(connection: Session) -> tuple[JsonObject, str | None]:
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
                job_id=row_string(job, "job_id"),
                revision=row_integer(job, "revision"),
                client_execution_id=row_string(
                    job,
                    "client_execution_id",
                ),
                fencing_token=row_integer(job, "fencing_token"),
            )
            return _encode_acquired(result), row_string(job, "job_id")

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

        def mutation(connection: Session) -> tuple[JsonObject, str | None]:
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
                row_string(current, "requested_device"),
                row_optional_string(current, "selected_device"),
                row_string(current, "operation"),
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
                and row_string(current, "execution_state")
                == ExecutionState.WAITING_INPUT.value
                and row_string(job, "execution_state")
                == ExecutionState.QUEUED.value
            )
            result = InputClosed(
                request_id=command.request_id,
                job_id=row_string(job, "job_id"),
                revision=row_integer(job, "revision"),
                input_state=InputState(row_string(job, "input_state")),
                input_revision=row_integer(job, "input_revision"),
                payload_count=row_integer(job, "payload_count"),
                total_rows=row_integer(job, "total_rows"),
                total_bytes=row_integer(job, "total_bytes"),
                manifest_sha256=row_string(job, "manifest_sha256"),
                execution_state=ExecutionState(
                    row_string(job, "execution_state")
                ),
            )
            return _encode_closed(result), row_string(job, "job_id")

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

        def mutation(connection: Session) -> tuple[JsonObject, str | None]:
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
                job_id=row_string(job, "job_id"),
                revision=row_integer(job, "revision"),
                input_state=InputState(row_string(job, "input_state")),
                execution_state=ExecutionState(
                    row_string(job, "execution_state")
                ),
            )
            return _encode_cancelled(result), row_string(job, "job_id")

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

    def _resolve_model(
        self,
        command: CreateJobCommand,
        connection: Session,
    ) -> PublishedModelRecord:
        model_ref = command.model_ref
        if model_ref is None:
            raise ValueError("job command requires a model reference")
        if command.operation == "predict" and command.model_selector == "alias":
            model = self.ledger.resolve_published_model_alias(
                command.owner_subject,
                model_ref,
                connection=connection,
                for_update=True,
            )
        else:
            model = self.ledger.get_published_model(
                model_ref,
                owner_subject=command.owner_subject,
                connection=connection,
                for_update=True,
            )
        if model is None:
            raise not_found("model generation not found")
        return model

def _require_request_hash(
    record: Mapping[str, object],
    request_hash: str,
) -> None:
    if row_string(record, "request_hash") != request_hash:
        raise conflict("idempotency key was used for a different request")


def _require_matching_identity(
    identity: Mapping[str, object],
    command: CreateJobCommand,
) -> None:
    if (
        row_string(identity, "owner_subject") != command.owner_subject
        or row_string(identity, "create_hash") != command.request_hash
    ):
        raise conflict("jobId has already been used for a different request")


def _require_current_fence(
    ledger: Ledger,
    command: CloseInputCommand | CancelJobCommand,
    connection: Session,
) -> None:
    job = ledger.get_job(
        command.job_id,
        owner_subject=command.owner_subject,
        connection=connection,
        for_update=True,
    )
    if job is None:
        raise not_found("job not found")
    if (
        row_string(job, "client_execution_id")
        != command.client_execution_id
        or row_integer(job, "fencing_token") != command.fencing_token
    ):
        raise ServiceError(
            ErrorCode.STALE_FENCE,
            "job ownership fence is stale",
        )


def _locations(
    values: tuple[tuple[str, str], ...],
) -> tuple[ArtifactLocation, ...]:
    return tuple(ArtifactLocation(*value) for value in values)


def _encode_created(result: JobCreated) -> JsonObject:
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
        "initialization": result.initialization,
        "limits": _encode_limits(result.limits),
    }


def _decode_created(document: JsonObject) -> JobCreated:
    if document.get("result_type") == "job_created":
        return JobCreated(
            request_id=_string(document, "request_id"),
            job_id=_string(document, "job_id"),
            operation=_string(document, "operation"),
            revision=_integer(document, "revision"),
            input_state=InputState(_string(document, "input_state")),
            input_revision=_integer(document, "input_revision"),
            next_input_ordinal=_integer(document, "next_input_ordinal"),
            execution_state=ExecutionState(
                _string(document, "execution_state")
            ),
            client_execution_id=_string(
                document,
                "client_execution_id",
            ),
            fencing_token=_integer(document, "fencing_token"),
            requested_device=_string(document, "requested_device"),
            selected_device=_optional_string(document, "selected_device"),
            resolved_model_ref=_optional_string(
                document,
                "resolved_model_ref",
            ),
            data_contract=_object(document, "data_contract"),
            ml_contract=_object(document, "ml_contract"),
            initialization=_optional_object(document, "initialization"),
            limits=_decode_limits(_object(document, "limits")),
        )
    ownership = _object(document, "ownership")
    input_document = _object(document, "input")
    execution = _object(document, "execution")
    device = _object(document, "device")
    return JobCreated(
        request_id=_string(document, "requestId"),
        job_id=_string(document, "jobId"),
        operation=_string(document, "operation"),
        revision=_integer(document, "revision"),
        input_state=InputState(_string(input_document, "state")),
        input_revision=_integer(input_document, "revision"),
        next_input_ordinal=_integer(input_document, "nextOrdinal"),
        execution_state=ExecutionState(_string(execution, "state")),
        client_execution_id=_string(ownership, "clientExecutionId"),
        fencing_token=_integer_text(ownership, "fencingToken"),
        requested_device=_string(device, "requested"),
        selected_device=_optional_string(device, "selected"),
        resolved_model_ref=_optional_string(document, "resolvedModelRef"),
        data_contract=_wire_data_contract(
            _object(document, "dataContract")
        ),
        ml_contract=_object(document, "mlContract"),
        initialization=_optional_object(document, "initialization"),
        limits=_wire_limits(_object(document, "limits")),
    )


def _encode_acquired(result: JobAcquired) -> JsonObject:
    return {
        "result_type": "job_acquired",
        "request_id": result.request_id,
        "job_id": result.job_id,
        "revision": result.revision,
        "client_execution_id": result.client_execution_id,
        "fencing_token": result.fencing_token,
    }


def _decode_acquired(document: JsonObject) -> JobAcquired:
    if document.get("result_type") == "job_acquired":
        return JobAcquired(
            request_id=_string(document, "request_id"),
            job_id=_string(document, "job_id"),
            revision=_integer(document, "revision"),
            client_execution_id=_string(
                document,
                "client_execution_id",
            ),
            fencing_token=_integer(document, "fencing_token"),
        )
    ownership = _object(document, "ownership")
    return JobAcquired(
        request_id=_string(document, "requestId"),
        job_id=_string(document, "jobId"),
        revision=_integer(document, "revision"),
        client_execution_id=_string(ownership, "clientExecutionId"),
        fencing_token=_integer_text(ownership, "fencingToken"),
    )


def _encode_closed(result: InputClosed) -> JsonObject:
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


def _decode_closed(document: JsonObject) -> InputClosed:
    if document.get("result_type") == "input_closed":
        return InputClosed(
            request_id=_string(document, "request_id"),
            job_id=_string(document, "job_id"),
            revision=_integer(document, "revision"),
            input_state=InputState(_string(document, "input_state")),
            input_revision=_integer(document, "input_revision"),
            payload_count=_integer(document, "payload_count"),
            total_rows=_integer(document, "total_rows"),
            total_bytes=_integer(document, "total_bytes"),
            manifest_sha256=_string(document, "manifest_sha256"),
            execution_state=ExecutionState(
                _string(document, "execution_state")
            ),
        )
    input_document = _object(document, "input")
    execution = _object(document, "execution")
    return InputClosed(
        request_id=_string(document, "requestId"),
        job_id=_string(document, "jobId"),
        revision=_integer(document, "revision"),
        input_state=InputState(_string(input_document, "state")),
        input_revision=_integer(input_document, "revision"),
        payload_count=_integer(input_document, "payloadCount"),
        total_rows=_integer(input_document, "totalRows"),
        total_bytes=_integer(input_document, "totalBytes"),
        manifest_sha256=_string(input_document, "manifestSha256"),
        execution_state=ExecutionState(_string(execution, "state")),
    )


def _encode_cancelled(result: JobCancelled) -> JsonObject:
    return {
        "result_type": "job_cancelled",
        "request_id": result.request_id,
        "job_id": result.job_id,
        "revision": result.revision,
        "input_state": result.input_state.value,
        "execution_state": result.execution_state.value,
    }


def _decode_cancelled(document: JsonObject) -> JobCancelled:
    if document.get("result_type") == "job_cancelled":
        return JobCancelled(
            request_id=_string(document, "request_id"),
            job_id=_string(document, "job_id"),
            revision=_integer(document, "revision"),
            input_state=InputState(_string(document, "input_state")),
            execution_state=ExecutionState(
                _string(document, "execution_state")
            ),
        )
    input_document = _object(document, "input")
    execution = _object(document, "execution")
    return JobCancelled(
        request_id=_string(document, "requestId"),
        job_id=_string(document, "jobId"),
        revision=_integer(document, "revision"),
        input_state=InputState(_string(input_document, "state")),
        execution_state=ExecutionState(_string(execution, "state")),
    )


def _encode_limits(limits: ServiceLimits) -> JsonObject:
    return {
        "max_message_bytes": limits.max_message_bytes,
        "target_batch_bytes": limits.target_batch_bytes,
        "max_batch_bytes": limits.max_batch_bytes,
        "max_payload_bytes": limits.max_payload_bytes,
        "max_rows_per_payload": limits.max_rows_per_payload,
        "max_payloads_per_job": limits.max_payloads_per_job,
        "max_job_bytes": limits.max_job_bytes,
        "max_active_jobs_per_subject": limits.max_active_jobs_per_subject,
        "max_page_items": limits.max_page_items,
        "input_idle_timeout_seconds": limits.input_idle_timeout_seconds,
    }


def _decode_limits(document: JsonObject) -> ServiceLimits:
    return ServiceLimits(
        max_message_bytes=_integer(document, "max_message_bytes"),
        target_batch_bytes=_integer(document, "target_batch_bytes"),
        max_batch_bytes=_integer(document, "max_batch_bytes"),
        max_payload_bytes=_integer(document, "max_payload_bytes"),
        max_rows_per_payload=_integer(document, "max_rows_per_payload"),
        max_payloads_per_job=_integer(document, "max_payloads_per_job"),
        max_job_bytes=_integer(document, "max_job_bytes"),
        max_active_jobs_per_subject=_integer(
            document,
            "max_active_jobs_per_subject",
        ),
        max_page_items=_integer(document, "max_page_items"),
        input_idle_timeout_seconds=_number(
            document,
            "input_idle_timeout_seconds",
        ),
    )


def _wire_limits(document: JsonObject) -> ServiceLimits:
    return ServiceLimits(
        max_message_bytes=_integer(document, "maxMessageBytes"),
        target_batch_bytes=_integer(document, "targetBatchBytes"),
        max_batch_bytes=_integer(document, "maxBatchBytes"),
        max_payload_bytes=_integer(document, "maxPayloadBytes"),
        max_rows_per_payload=_integer(document, "maxRowsPerPayload"),
        max_payloads_per_job=_integer(document, "maxPayloadsPerJob"),
        max_job_bytes=_integer(document, "maxJobBytes"),
        max_active_jobs_per_subject=_integer(
            document,
            "maxActiveJobsPerSubject",
        ),
        max_page_items=_integer(document, "maxPageItems"),
        input_idle_timeout_seconds=_number(
            document,
            "inputIdleTimeoutSeconds",
        ),
    )


def _wire_data_contract(document: JsonObject) -> JsonObject:
    return {
        "id": _string(document, "id"),
        "version": _integer(document, "version"),
        "profile": _string(document, "profile"),
        "data_contract_sha256": _string(
            document,
            "dataContractSha256",
        ),
        "seq_len": _integer(document, "seqLen"),
        "feature_dim": _integer(document, "featureDim"),
        "target_schema_id": _string(document, "targetSchemaId"),
    }


def _string(document: JsonObject, key: str) -> str:
    value = _value(document, key)
    if not isinstance(value, str):
        raise ValueError(f"stored result field {key} must be a string")
    return value


def _optional_string(document: JsonObject, key: str) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"stored result field {key} must be a string or null"
        )
    return value


def _integer(document: JsonObject, key: str) -> int:
    value = _value(document, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"stored result field {key} must be an integer")
    return value


def _integer_text(document: JsonObject, key: str) -> int:
    value = _value(document, key)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(
            f"stored result field {key} must encode an integer"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"stored result field {key} must encode an integer"
        ) from exc


def _number(document: JsonObject, key: str) -> float:
    value = _value(document, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"stored result field {key} must be numeric")
    return float(value)


def _object(document: JsonObject, key: str) -> JsonObject:
    value = _value(document, key)
    if not isinstance(value, dict):
        raise ValueError(f"stored result field {key} must be an object")
    return value


def _optional_object(document: JsonObject, key: str) -> JsonObject | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"stored result field {key} must be an object or null"
        )
    return value


def _value(document: JsonObject, key: str) -> JsonValue:
    if key not in document:
        raise ValueError(f"stored result field {key} is missing")
    return document[key]


__all__ = ["JobActionNames", "PostgresJobLifecycle"]
