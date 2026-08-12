import pyarrow

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
from app.service.adapters.inbound.flight.contract import (
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
from app.service.application.job_models import (
    AcquireJobCommand,
    CancelJobCommand,
    CloseInputCommand,
    CreateJobCommand,
    DescribeModelQuery,
    GetJobStatusQuery,
    ListJobInputsQuery,
    ListJobOutputsQuery,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.version import __version__


class JobCoordinator:
    def __init__(
        self,
        *,
        create_job,
        acquire_job,
        close_input,
        cancel_job,
        get_status,
        list_inputs,
        list_outputs,
        describe_model,
        service_status,
        availability,
    ):
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

    def dispatch(self, action: str, owner: str, request: dict, document: dict) -> bytes:
        if action == CAPABILITIES_ACTION:
            result = self.capabilities(request["request_id"])
        elif action == HEALTH_ACTION:
            result = self.health(request["request_id"])
        elif action == CREATE_ACTION:
            result = present_job_created(
                self._create_job.create(
                    _create_command(owner, request, document)
                )
            )
        elif action == ACQUIRE_ACTION:
            result = present_job_acquired(
                self._acquire_job.acquire(
                    _acquire_command(owner, request, document)
                )
            )
        elif action == STATUS_ACTION:
            result = present_job_status(
                self._get_status.execute(
                    GetJobStatusQuery(
                        owner_subject=owner,
                        request_id=request["request_id"],
                        job_id=request["job_id"],
                    )
                )
            )
        elif action == INPUTS_LIST_ACTION:
            result = present_job_inputs(
                self._list_inputs.execute(
                    ListJobInputsQuery(
                        owner_subject=owner,
                        request_id=request["request_id"],
                        job_id=request["job_id"],
                        after_revision=request["after_revision"],
                        snapshot_revision=request["snapshot_revision"],
                        cursor=request["cursor"],
                        limit=request["limit"],
                    )
                )
            )
        elif action == INPUT_CLOSE_ACTION:
            result = present_input_closed(
                self._close_input.close(
                    _close_command(owner, request, document)
                )
            )
        elif action == OUTPUTS_LIST_ACTION:
            result = present_job_outputs(
                self._list_outputs.execute(
                    ListJobOutputsQuery(
                        owner_subject=owner,
                        request_id=request["request_id"],
                        job_id=request["job_id"],
                        cursor=request["cursor"],
                        limit=request["limit"],
                    )
                )
            )
        elif action == CANCEL_ACTION:
            result = present_job_cancelled(
                self._cancel_job.cancel(
                    _cancel_command(owner, request, document)
                )
            )
        elif action == MODEL_DESCRIBE_ACTION:
            result = present_model_description(
                self._describe_model.execute(
                    DescribeModelQuery(
                        owner_subject=owner,
                        request_id=request["request_id"],
                        model_selector=(
                            "alias"
                            if request["model_selector"] == "modelAlias"
                            else "reference"
                        ),
                        model_ref=request["model_ref"],
                    )
                )
            )
        else:
            raise ServiceError(
                ErrorCode.INVALID_ARGUMENT,
                f"unsupported action: {action}",
            )
        return encode_document(result)

    def capabilities(self, request_id: str) -> dict:
        capabilities = self._service_status.capabilities()
        inventory = capabilities.device_inventory
        return response_document(
            request_id,
            protocolVersions=[CONTRACT_VERSION],
            service={
                "name": "transformer-flight",
                "version": __version__,
                "pyarrowVersion": pyarrow.__version__,
                "torchVersion": inventory.torch_version,
            },
            schemaIds={
                "fitInput": FIT_SCHEMA_ID,
                "predictInput": PREDICT_SCHEMA_ID,
                "predictionOutput": PREDICTION_SCHEMA_ID,
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

    def health(self, request_id: str) -> dict:
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


def _create_command(owner: str, request: dict, document: dict) -> CreateJobCommand:
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
        data_contract=request["data_contract"],
        model_label=request.get("model_label"),
        model_selector=(
            None
            if request.get("model_selector") is None
            else (
                "alias"
                if request["model_selector"] == "modelAlias"
                else "reference"
            )
        ),
        model_ref=request.get("model_ref"),
        model_config=request.get("model_config"),
        training_config=request.get("train_config"),
    )


def _acquire_command(owner: str, request: dict, document: dict) -> AcquireJobCommand:
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


def _close_command(owner: str, request: dict, document: dict) -> CloseInputCommand:
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


def _cancel_command(owner: str, request: dict, document: dict) -> CancelJobCommand:
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
