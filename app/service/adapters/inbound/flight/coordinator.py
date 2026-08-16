from typing import cast

import pyarrow

from app.contracts.json_types import JsonObject
from app.contracts.worker.v4.objective import (
    CHECKPOINT_FORMAT,
    OBJECTIVE_ID,
    PREDICTION_SCHEMA_ID as ML_PREDICTION_SCHEMA_ID,
    TARGET_SCHEMA_ID,
    TARGET_WIDTH,
)
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CONTRACT_VERSION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MODEL_DESCRIBE_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_SCHEMA_ID,
    PREDICTION_SCHEMA_ID,
    STATUS_ACTION,
)
from app.service.adapters.inbound.flight.documents import (
    canonical_request_hash,
    encode_document,
    response_document,
)
from app.service.adapters.inbound.flight.presentation import (
    limits_to_api,
    present_input_closed,
    present_job_acquired,
    present_job_cancelled,
    present_job_created,
    present_job_inputs,
    present_job_outputs,
    present_job_status,
    present_model_description,
)
from app.service.adapters.inbound.flight.validation import (
    AcquireRequestFields,
    CancelRequestFields,
    CreateRequestFields,
    InputCloseRequestFields,
    InputsListRequestFields,
    ModelDescribeRequestFields,
    OutputsListRequestFields,
    RequestIdFields,
    StatusRequestFields,
    ValidatedActionRequest,
)
from app.service.application.commands.jobs import (
    AcquireJobAction,
    CancelJobAction,
    CreateJobAction,
    InputCloseAction,
)
from app.service.application.messages.jobs import (
    AcquireJobCommand,
    CancelJobCommand,
    CloseInputCommand,
    CreateJobCommand,
    DescribeModelQuery,
    GetJobStatusQuery,
    ListJobInputsQuery,
    ListJobOutputsQuery,
)
from app.service.application.queries.service import (
    ServiceAvailability,
    ServiceStatusQuery,
)
from app.service.application.queries.status import (
    DescribeModel,
    GetJobStatus,
    ListJobInputs,
    ListJobOutputs,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.version import __version__


class JobCoordinator:
    def __init__(
        self,
        *,
        create_job: CreateJobAction,
        acquire_job: AcquireJobAction,
        close_input: InputCloseAction,
        cancel_job: CancelJobAction,
        get_status: GetJobStatus,
        list_inputs: ListJobInputs,
        list_outputs: ListJobOutputs,
        describe_model: DescribeModel,
        service_status: ServiceStatusQuery,
        availability: ServiceAvailability,
    ) -> None:
        self._create_job = create_job
        self._acquire_job = acquire_job
        self._close_input = close_input
        self._cancel_job = cancel_job
        self._get_status = get_status
        self._list_inputs = list_inputs
        self._list_outputs = list_outputs
        self._describe_model = describe_model
        self._service_status = service_status
        self._availability = availability

    def dispatch(
        self,
        action: str,
        owner: str,
        request: ValidatedActionRequest,
        document: JsonObject,
    ) -> bytes:
        if action == CAPABILITIES_ACTION:
            fields = cast(RequestIdFields, request)
            result = self.capabilities(fields["request_id"])
        elif action == HEALTH_ACTION:
            fields = cast(RequestIdFields, request)
            result = self.health(fields["request_id"])
        elif action == CREATE_ACTION:
            fields = cast(CreateRequestFields, request)
            result = present_job_created(
                self._create_job.create(
                    _create_command(owner, fields, document)
                )
            )
        elif action == ACQUIRE_ACTION:
            fields = cast(AcquireRequestFields, request)
            result = present_job_acquired(
                self._acquire_job.acquire(
                    _acquire_command(owner, fields, document)
                )
            )
        elif action == STATUS_ACTION:
            fields = cast(StatusRequestFields, request)
            result = present_job_status(
                self._get_status.execute(
                    GetJobStatusQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        job_id=fields["job_id"],
                    )
                )
            )
        elif action == INPUTS_LIST_ACTION:
            fields = cast(InputsListRequestFields, request)
            result = present_job_inputs(
                self._list_inputs.execute(
                    ListJobInputsQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        job_id=fields["job_id"],
                        after_revision=fields["after_revision"],
                        snapshot_revision=fields["snapshot_revision"],
                        cursor=fields["cursor"],
                        limit=fields["limit"],
                    )
                )
            )
        elif action == INPUT_CLOSE_ACTION:
            fields = cast(InputCloseRequestFields, request)
            result = present_input_closed(
                self._close_input.close(
                    _close_command(owner, fields, document)
                )
            )
        elif action == OUTPUTS_LIST_ACTION:
            fields = cast(OutputsListRequestFields, request)
            result = present_job_outputs(
                self._list_outputs.execute(
                    ListJobOutputsQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        job_id=fields["job_id"],
                        cursor=fields["cursor"],
                        limit=fields["limit"],
                    )
                )
            )
        elif action == CANCEL_ACTION:
            fields = cast(CancelRequestFields, request)
            result = present_job_cancelled(
                self._cancel_job.cancel(
                    _cancel_command(owner, fields, document)
                )
            )
        elif action == MODEL_DESCRIBE_ACTION:
            fields = cast(ModelDescribeRequestFields, request)
            result = present_model_description(
                self._describe_model.execute(
                    DescribeModelQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        model_selector=(
                            "alias"
                            if fields["model_selector"] == "modelAlias"
                            else "reference"
                        ),
                        model_ref=fields["model_ref"],
                    )
                )
            )
        else:
            raise ServiceError(
                ErrorCode.INVALID_ARGUMENT,
                f"unsupported action: {action}",
            )
        return encode_document(result)

    def capabilities(self, request_id: str) -> JsonObject:
        capabilities = self._service_status.capabilities()
        inventory = capabilities.device_inventory
        return response_document(
            request_id,
            protocolVersions=[CONTRACT_VERSION],
            service={
                "name": "transformer-flight",
                "version": __version__,
                "pyarrowVersion": str(vars(pyarrow)["__version__"]),
                "torchVersion": inventory.torch_version,
            },
            schemaIds={
                "fitInput": FIT_SCHEMA_ID,
                "predictInput": PREDICT_SCHEMA_ID,
                "predictionOutput": PREDICTION_SCHEMA_ID,
            },
            mlContract={
                "targetSchemaId": TARGET_SCHEMA_ID,
                "predictionSchemaId": ML_PREDICTION_SCHEMA_ID,
                "objectiveId": OBJECTIVE_ID,
                "checkpointFormat": CHECKPOINT_FORMAT,
                "targetWidth": TARGET_WIDTH,
                "predictionSpace": "target",
                "objectiveConfigSchemaVersion": 1,
            },
            limits=limits_to_api(capabilities.limits),
            devices={
                "cpu": {"available": True},
                "cuda": {
                    "available": inventory.cuda_capacity > 0,
                    "deviceCount": inventory.device_count,
                    "quarantinedCount": inventory.quarantined_count,
                    "runtimeVersion": inventory.runtime_version,
                },
            },
            queue={
                "cpuCapacity": capabilities.cpu_capacity,
                "cudaCapacity": inventory.cuda_capacity,
                "singleInstance": True,
            },
            supportedOperations=["fit", "predict"],
            features={
                "doExchange": False,
                "pollFlightInfo": False,
                "durableStreamingInput": True,
                "clientGeneratedJobId": True,
                "crossSystemFencing": True,
                "revisionPagination": True,
                "resumableFit": True,
                "recoveryBoundary": "globalEpoch",
                "deviceAwareCuda": True,
            },
        )

    def health(self, request_id: str) -> JsonObject:
        health = self._service_status.health()
        inventory = health.device_inventory
        return response_document(
            request_id,
            live=True,
            ready=health.ready,
            draining=health.draining,
            ledger={"available": health.ledger_available},
            cuda={
                "available": inventory.cuda_capacity > 0,
                "deviceCount": inventory.device_count,
                "quarantinedCount": inventory.quarantined_count,
            },
            storage={
                "runtime": {"freeBytes": health.runtime_storage.free},
                "recovery": {"freeBytes": health.recovery_storage.free},
            },
            metrics=health.metrics,
        )

    def set_draining(self, value: bool = True) -> None:
        self._availability.set_draining(value)


def _create_command(
    owner: str,
    request: CreateRequestFields,
    document: JsonObject,
) -> CreateJobCommand:
    wire_model_selector = request.get("model_selector")
    return CreateJobCommand(
        owner_subject=owner,
        request_id=request["request_id"],
        idempotency_key=request["idempotency_key"],
        request_hash=canonical_request_hash(document),
        job_id=request["job_id"],
        client_execution_id=request["client_execution_id"],
        operation=request["operation"],
        requested_device=request["device"],
        prediction_column=request["prediction_column"],
        data_contract=cast(JsonObject, dict(request["data_contract"])),
        ml_contract=cast(JsonObject, dict(request["ml_contract"])),
        model_label=request.get("model_label"),
        model_selector=(
            None
            if wire_model_selector is None
            else (
                "alias"
                if wire_model_selector == "modelAlias"
                else "reference"
            )
        ),
        model_ref=request.get("model_ref"),
        model_config=request.get("model_config"),
        training_config=request.get("train_config"),
    )


def _acquire_command(
    owner: str,
    request: AcquireRequestFields,
    document: JsonObject,
) -> AcquireJobCommand:
    return AcquireJobCommand(
        owner_subject=owner,
        request_id=request["request_id"],
        idempotency_key=request["idempotency_key"],
        request_hash=canonical_request_hash(document),
        job_id=request["job_id"],
        previous_client_execution_id=request["previous_client_execution_id"],
        expected_fencing_token=request["expected_fencing_token"],
        client_execution_id=request["client_execution_id"],
    )


def _close_command(
    owner: str,
    request: InputCloseRequestFields,
    document: JsonObject,
) -> CloseInputCommand:
    return CloseInputCommand(
        owner_subject=owner,
        request_id=request["request_id"],
        idempotency_key=request["idempotency_key"],
        request_hash=canonical_request_hash(document),
        job_id=request["job_id"],
        client_execution_id=request["client_execution_id"],
        fencing_token=request["fencing_token"],
        payload_count=request["payload_count"],
        total_rows=request["total_rows"],
        total_bytes=request["total_bytes"],
        manifest_sha256=request["manifest_sha256"],
    )


def _cancel_command(
    owner: str,
    request: CancelRequestFields,
    document: JsonObject,
) -> CancelJobCommand:
    return CancelJobCommand(
        owner_subject=owner,
        request_id=request["request_id"],
        idempotency_key=request["idempotency_key"],
        request_hash=canonical_request_hash(document),
        job_id=request["job_id"],
        client_execution_id=request["client_execution_id"],
        fencing_token=request["fencing_token"],
    )


__all__ = ["JobCoordinator"]
