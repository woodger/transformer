from typing import cast

import pyarrow

from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v2 import (
    MAX_RESPONSE_BYTES as MODEL_CATALOG_MAX_RESPONSE_BYTES,
    catalog_capabilities,
    validate_catalog_document,
)
from app.contracts.semantic.v2 import semantic_capabilities
from app.contracts.training_telemetry.v2 import training_telemetry_capabilities
from app.contracts.worker.v13.constants import (
    CHECKPOINT_FORMAT,
    CONTRACT_VERSION as WORKER_CONTRACT_VERSION,
    RECOVERY_FORMAT,
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
    MODEL_CATALOG_DETAIL_ACTION,
    MODEL_CATALOG_LIST_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_SCHEMA_ID,
    PREDICTION_SCHEMA_ID,
    STATUS_ACTION,
    TRAINING_TELEMETRY_GRADIENT_ACTION,
    TRAINING_TELEMETRY_REPORT_ACTION,
)
from app.service.adapters.inbound.flight.devices import device_from_api
from app.service.adapters.inbound.flight.documents import (
    canonical_request_hash,
    encode_document,
    response_document,
)
from app.service.adapters.inbound.flight.model_catalog import (
    artifact_error,
    catalog_error,
    expired_catalog_cursor,
    invalid_catalog_cursor,
    model_not_found,
    registry_unavailable,
)
from app.service.adapters.inbound.flight.presentation import (
    limits_to_api,
    present_catalog_model_detail,
    present_catalog_models_page,
    present_input_closed,
    present_job_acquired,
    present_job_cancelled,
    present_job_created,
    present_job_inputs,
    present_job_outputs,
    present_job_status,
)
from app.service.adapters.inbound.flight.training_telemetry import (
    expired_telemetry_cursor,
    invalid_telemetry_cursor,
    invalidated_telemetry_cursor,
    present_gradient_interactions,
    present_training_telemetry_report,
    telemetry_backend_unavailable,
    telemetry_integrity_failed,
    telemetry_model_not_found,
    telemetry_snapshot_capacity_exhausted,
    telemetry_stored_metadata_invalid,
    training_telemetry_response,
)
from app.service.adapters.inbound.flight.validation import (
    AcquireRequestFields,
    CancelRequestFields,
    CreateRequestFields,
    InputCloseRequestFields,
    InputsListRequestFields,
    ModelCatalogDetailRequestFields,
    ModelCatalogListRequestFields,
    OutputsListRequestFields,
    RequestIdFields,
    StatusRequestFields,
    TrainingTelemetryGradientRequestFields,
    TrainingTelemetryReportRequestFields,
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
    GetJobStatusQuery,
    ListJobInputsQuery,
    ListJobOutputsQuery,
)
from app.service.application.messages.model_catalog import (
    GetCatalogModelQuery,
    ListCatalogModelsQuery,
)
from app.service.application.messages.training_telemetry import (
    GetGradientInteractionsQuery,
    GetTrainingTelemetryReportQuery,
)
from app.service.application.ports.model_catalog import (
    CatalogArtifactVerificationError,
    CatalogModelNotFound,
    ModelCatalogStoreUnavailable,
)
from app.service.application.ports.training_telemetry import (
    TrainingTelemetryBackendUnavailable,
    TrainingTelemetryIntegrityError,
    TrainingTelemetryStoredMetadataError,
)
from app.service.application.queries.model_catalog import (
    GetCatalogModel,
    ListCatalogModels,
)
from app.service.application.queries.service import (
    ServiceAvailability,
    ServiceStatusQuery,
)
from app.service.application.queries.status import (
    GetJobStatus,
    ListJobInputs,
    ListJobOutputs,
)
from app.service.application.queries.training_telemetry import (
    GetGradientInteractions,
    GetTrainingTelemetryReport,
)
from app.service.application.services.model_catalog_cursor import (
    ExpiredCatalogCursor,
    InvalidCatalogCursor,
)
from app.service.application.services.training_telemetry_cursor import (
    ExpiredTrainingTelemetryCursor,
    InvalidatedTrainingTelemetryCursor,
    InvalidTrainingTelemetryCursor,
)
from app.service.application.services.training_telemetry_snapshot import (
    TrainingTelemetrySnapshotCapacityExhausted,
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
        list_catalog_models: ListCatalogModels,
        get_catalog_model: GetCatalogModel,
        service_status: ServiceStatusQuery,
        availability: ServiceAvailability,
        get_training_telemetry_report: GetTrainingTelemetryReport | None = None,
        get_gradient_interactions: GetGradientInteractions | None = None,
    ) -> None:
        self._create_job = create_job
        self._acquire_job = acquire_job
        self._close_input = close_input
        self._cancel_job = cancel_job
        self._get_status = get_status
        self._list_inputs = list_inputs
        self._list_outputs = list_outputs
        self._list_catalog_models = list_catalog_models
        self._get_catalog_model = get_catalog_model
        self._service_status = service_status
        self._availability = availability
        self._get_training_telemetry_report = get_training_telemetry_report
        self._get_gradient_interactions = get_gradient_interactions

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
        elif action == MODEL_CATALOG_LIST_ACTION:
            fields = cast(ModelCatalogListRequestFields, request)
            try:
                result = present_catalog_models_page(
                    self._list_catalog_models.execute(
                        ListCatalogModelsQuery(
                            owner_subject=owner,
                            request_id=fields["request_id"],
                            page_size=fields["page_size"],
                            cursor=fields["cursor"],
                        )
                    )
                )
            except ExpiredCatalogCursor as exc:
                raise expired_catalog_cursor() from exc
            except InvalidCatalogCursor as exc:
                raise invalid_catalog_cursor() from exc
            except ModelCatalogStoreUnavailable as exc:
                raise registry_unavailable() from exc
            return _catalog_response(result, "list-result")
        elif action == MODEL_CATALOG_DETAIL_ACTION:
            fields = cast(ModelCatalogDetailRequestFields, request)
            try:
                result = present_catalog_model_detail(
                    self._get_catalog_model.execute(
                        GetCatalogModelQuery(
                            owner_subject=owner,
                            request_id=fields["request_id"],
                            model_ref=fields["model_ref"],
                        )
                    )
                )
            except CatalogModelNotFound as exc:
                raise model_not_found(exc.model_ref) from exc
            except CatalogArtifactVerificationError as exc:
                raise artifact_error(exc) from exc
            except ModelCatalogStoreUnavailable as exc:
                raise registry_unavailable() from exc
            return _catalog_response(result, "detail-result")
        elif action == TRAINING_TELEMETRY_REPORT_ACTION:
            fields = cast(TrainingTelemetryReportRequestFields, request)
            if self._get_training_telemetry_report is None:
                raise RuntimeError("training telemetry report query is unavailable")
            try:
                result = present_training_telemetry_report(
                    self._get_training_telemetry_report.execute(
                        GetTrainingTelemetryReportQuery(
                            owner_subject=owner,
                            request_id=fields["request_id"],
                            model_ref=fields["model_ref"],
                            page_size=fields["page_size"],
                            cursor=fields["cursor"],
                        )
                    )
                )
            except CatalogModelNotFound as exc:
                raise telemetry_model_not_found(exc.model_ref) from exc
            except InvalidTrainingTelemetryCursor as exc:
                raise invalid_telemetry_cursor() from exc
            except ExpiredTrainingTelemetryCursor as exc:
                raise expired_telemetry_cursor() from exc
            except InvalidatedTrainingTelemetryCursor as exc:
                raise invalidated_telemetry_cursor() from exc
            except TrainingTelemetrySnapshotCapacityExhausted as exc:
                raise telemetry_snapshot_capacity_exhausted(
                    exc.operation
                ) from exc
            except TrainingTelemetryStoredMetadataError as exc:
                raise telemetry_stored_metadata_invalid(
                    fields["model_ref"], exc.path
                ) from exc
            except TrainingTelemetryIntegrityError as exc:
                raise telemetry_integrity_failed(
                    fields["model_ref"], exc.path
                ) from exc
            except (TrainingTelemetryBackendUnavailable, ModelCatalogStoreUnavailable) as exc:
                raise telemetry_backend_unavailable(fields["model_ref"]) from exc
            return training_telemetry_response(result, "report-result")
        elif action == TRAINING_TELEMETRY_GRADIENT_ACTION:
            fields = cast(TrainingTelemetryGradientRequestFields, request)
            if self._get_gradient_interactions is None:
                raise RuntimeError("gradient interaction query is unavailable")
            try:
                result = present_gradient_interactions(
                    self._get_gradient_interactions.execute(
                        GetGradientInteractionsQuery(
                            owner_subject=owner,
                            request_id=fields["request_id"],
                            model_ref=fields["model_ref"],
                            epoch=fields["epoch"],
                            page_size=fields["page_size"],
                            cursor=fields["cursor"],
                        )
                    )
                )
            except CatalogModelNotFound as exc:
                raise telemetry_model_not_found(exc.model_ref) from exc
            except InvalidTrainingTelemetryCursor as exc:
                raise invalid_telemetry_cursor() from exc
            except ExpiredTrainingTelemetryCursor as exc:
                raise expired_telemetry_cursor() from exc
            except InvalidatedTrainingTelemetryCursor as exc:
                raise invalidated_telemetry_cursor() from exc
            except TrainingTelemetrySnapshotCapacityExhausted as exc:
                raise telemetry_snapshot_capacity_exhausted(
                    exc.operation
                ) from exc
            except TrainingTelemetryStoredMetadataError as exc:
                raise telemetry_stored_metadata_invalid(
                    fields["model_ref"], exc.path
                ) from exc
            except TrainingTelemetryIntegrityError as exc:
                raise telemetry_integrity_failed(
                    fields["model_ref"], exc.path
                ) from exc
            except (TrainingTelemetryBackendUnavailable, ModelCatalogStoreUnavailable) as exc:
                raise telemetry_backend_unavailable(fields["model_ref"]) from exc
            return training_telemetry_response(
                result,
                "gradient-interactions-result",
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
            sourceEncodings=["indexedFeatureBlocks"],
            workerProtocolVersion=WORKER_CONTRACT_VERSION,
            checkpointFormat=CHECKPOINT_FORMAT,
            recoveryFormat=RECOVERY_FORMAT,
            metricsFormats=[
                "transformer.fit-run-summary.v6",
                "transformer.training-metrics.v6",
            ],
            semantic=semantic_capabilities(),
            modelCatalog=catalog_capabilities(),
            trainingTelemetry=training_telemetry_capabilities(),
            limits=limits_to_api(capabilities.limits),
            devices={
                "cpu": {"available": True},
                "gpu": {
                    "available": inventory.cuda_capacity > 0,
                    "deviceCount": inventory.device_count,
                    "quarantinedCount": inventory.quarantined_count,
                },
            },
            queue={
                "cpuCapacity": capabilities.cpu_capacity,
                "gpuCapacity": inventory.cuda_capacity,
                "singleInstance": True,
            },
            supportedOperations=["fit", "predict"],
            fitInitializations=[
                "publishedModel",
                "random",
            ],
            features={
                "doExchange": False,
                "pollFlightInfo": False,
                "durableStreamingInput": True,
                "clientGeneratedJobId": True,
                "crossSystemFencing": True,
                "revisionPagination": True,
                "resumableFit": True,
                "recoveryBoundary": "globalEpoch",
                "deviceAwareGpu": True,
                "structuredErrorDetails": True,
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
            gpu={
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


def _catalog_response(document: JsonObject, schema_name: str) -> bytes:
    validate_catalog_document(document, schema_name)
    encoded = encode_document(document)
    if len(encoded) > MODEL_CATALOG_MAX_RESPONSE_BYTES:
        raise catalog_error(
            ErrorCode.RESOURCE_EXHAUSTED,
            "CATALOG_RESPONSE_BUDGET_EXCEEDED",
            "model catalog response exceeds the configured budget",
            maxBytes=MODEL_CATALOG_MAX_RESPONSE_BYTES,
        )
    return encoded


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
        requested_device=device_from_api(request["device"]),
        prediction_column=request["prediction_column"],
        source_encoding=dict(request["source_encoding"]),
        data_contract=cast(JsonObject, dict(request["data_contract"])),
        model_contract=dict(request["model_contract"]),
        semantic_digests=dict(request["semantic_digests"]),
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
        initialization_source=request.get("initialization_source"),
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
        total_chunks=request["total_chunks"],
        total_rows=request["total_rows"],
        total_native_rows=tuple(request["total_native_rows"]),
        range_count=request["range_count"],
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
