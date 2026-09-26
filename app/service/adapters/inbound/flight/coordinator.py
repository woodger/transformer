from collections.abc import Callable
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v7 import (
    MAX_RESPONSE_BYTES as MODEL_CATALOG_MAX_RESPONSE_BYTES,
    validate_catalog_document,
)
from app.contracts.semantic.v5 import semantic_capabilities
from app.contracts.semantic.v5.constants import ENCODER_NORMALIZATION_ORDERS
from app.contracts.target_head_diagnostics.v5.constants import (
    CONTRACT_NAME as TARGET_HEAD_DIAGNOSTICS_CONTRACT_NAME,
    CONTRACT_REVISION as TARGET_HEAD_DIAGNOSTICS_CONTRACT_REVISION,
    CURSOR_TTL_SECONDS as TARGET_HEAD_DIAGNOSTICS_CURSOR_TTL_SECONDS,
    MAX_COMMITTED_ARTIFACT_ROWS,
    MAX_EPOCH_PAGE_SIZE as TARGET_HEAD_DIAGNOSTICS_MAX_EPOCH_PAGE_SIZE,
    MAX_RESPONSE_BYTES as TARGET_HEAD_DIAGNOSTICS_MAX_RESPONSE_BYTES,
    MAX_RETAINED_SNAPSHOT_BYTES as TARGET_HEAD_DIAGNOSTICS_MAX_SNAPSHOT_BYTES,
    MAX_RETAINED_SNAPSHOT_COUNT as TARGET_HEAD_DIAGNOSTICS_MAX_SNAPSHOT_COUNT,
    MAX_RETAINED_SNAPSHOT_TOTAL_BYTES as TARGET_HEAD_DIAGNOSTICS_MAX_SNAPSHOT_TOTAL_BYTES,
)
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    FIT_CREATE_ACTION,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MODEL_CATALOG_DETAIL_ACTION,
    MODEL_CATALOG_LIST_ACTION,
    MODEL_TOPOLOGY_DETAIL_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_CREATE_ACTION,
    STATUS_ACTION,
    TARGET_HEAD_DIAGNOSTICS_REPORT_ACTION,
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
from app.service.adapters.inbound.flight.model_topology import (
    model_topology_response,
    model_topology_unavailable,
    present_model_topology,
    topology_invalid,
    topology_model_not_found,
    topology_registry_unavailable,
    topology_response_budget_exceeded,
    topology_stored_metadata_invalid,
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
from app.service.adapters.inbound.flight.target_head_diagnostics import (
    expired_target_head_diagnostics_cursor,
    invalid_target_head_diagnostics_cursor,
    invalidated_target_head_diagnostics_cursor,
    target_head_diagnostics_backend_unavailable,
    target_head_diagnostics_integrity_failed,
    target_head_diagnostics_model_not_found,
    target_head_diagnostics_query_unavailable,
    target_head_diagnostics_response,
    target_head_diagnostics_snapshot_capacity_exhausted,
    target_head_diagnostics_stored_metadata_invalid,
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
    ModelTopologyDetailRequestFields,
    OutputsListRequestFields,
    RequestIdFields,
    StatusRequestFields,
    TargetHeadDiagnosticsReportRequestFields,
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
from app.service.application.messages.model_topology import (
    GetModelTopologyQuery,
)
from app.service.application.messages.target_head_diagnostics import (
    GetTargetHeadDiagnosticsReportQuery,
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
from app.service.application.queries.model_topology import GetModelTopology
from app.service.application.queries.service import (
    ServiceAvailability,
    ServiceStatusQuery,
)
from app.service.application.queries.status import (
    GetJobStatus,
    ListJobInputs,
    ListJobOutputs,
)
from app.service.application.queries.target_head_diagnostics import (
    GetTargetHeadDiagnosticsReport,
)
from app.service.application.queries.training_telemetry import (
    GetGradientInteractions,
    GetTrainingTelemetryReport,
)
from app.service.application.services.model_catalog_cursor import (
    ExpiredCatalogCursor,
    InvalidCatalogCursor,
)
from app.service.application.services.target_head_diagnostics_cursor import (
    TargetHeadDiagnosticsCursorError,
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
        get_model_topology: GetModelTopology | None = None,
        service_status: ServiceStatusQuery,
        availability: ServiceAvailability,
        get_training_telemetry_report: GetTrainingTelemetryReport | None = None,
        get_gradient_interactions: GetGradientInteractions | None = None,
        get_target_head_diagnostics_report: (
            GetTargetHeadDiagnosticsReport | None
        ) = None,
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
        self._get_model_topology = get_model_topology
        self._service_status = service_status
        self._availability = availability
        self._get_training_telemetry_report = get_training_telemetry_report
        self._get_gradient_interactions = get_gradient_interactions
        self._get_target_head_diagnostics_report = (
            get_target_head_diagnostics_report
        )

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
        elif action in (FIT_CREATE_ACTION, PREDICT_CREATE_ACTION):
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
            return self._catalog_list_response(owner, fields)
        elif action == MODEL_CATALOG_DETAIL_ACTION:
            fields = cast(ModelCatalogDetailRequestFields, request)
            return self._catalog_detail_response(owner, fields)
        elif action == MODEL_TOPOLOGY_DETAIL_ACTION:
            fields = cast(ModelTopologyDetailRequestFields, request)
            return self._model_topology_detail_response(owner, fields)
        elif action == TRAINING_TELEMETRY_REPORT_ACTION:
            fields = cast(TrainingTelemetryReportRequestFields, request)
            return self._training_telemetry_report_response(owner, fields)
        elif action == TRAINING_TELEMETRY_GRADIENT_ACTION:
            fields = cast(TrainingTelemetryGradientRequestFields, request)
            return self._gradient_interactions_response(owner, fields)
        elif action == TARGET_HEAD_DIAGNOSTICS_REPORT_ACTION:
            fields = cast(TargetHeadDiagnosticsReportRequestFields, request)
            return self._target_head_diagnostics_report_response(owner, fields)
        else:
            raise ServiceError(
                ErrorCode.INVALID_ARGUMENT,
                f"unsupported action: {action}",
            )
        return encode_document(result)

    def _catalog_list_response(
        self,
        owner: str,
        fields: ModelCatalogListRequestFields,
    ) -> bytes:
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

    def _catalog_detail_response(
        self,
        owner: str,
        fields: ModelCatalogDetailRequestFields,
    ) -> bytes:
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

    def _model_topology_detail_response(
        self,
        owner: str,
        fields: ModelTopologyDetailRequestFields,
    ) -> bytes:
        if self._get_model_topology is None:
            raise model_topology_unavailable()
        try:
            result = present_model_topology(
                self._get_model_topology.execute(
                    GetModelTopologyQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        model_ref=fields["model_ref"],
                    )
                )
            )
            return model_topology_response(result)
        except CatalogModelNotFound as exc:
            raise topology_model_not_found(exc.model_ref) from exc
        except ModelCatalogStoreUnavailable as exc:
            raise topology_registry_unavailable() from exc
        except ServiceError as exc:
            if _stored_metadata_error(exc):
                raise topology_stored_metadata_invalid(
                    fields["model_ref"],
                    _error_path(exc),
                ) from exc
            raise
        except OverflowError as exc:
            raise topology_response_budget_exceeded(
                fields["model_ref"]
            ) from exc
        except ValueError as exc:
            raise topology_invalid(fields["model_ref"], "") from exc

    def _training_telemetry_report_response(
        self,
        owner: str,
        fields: TrainingTelemetryReportRequestFields,
    ) -> bytes:
        report_query = self._get_training_telemetry_report
        if report_query is None:
            raise RuntimeError("training telemetry report query is unavailable")
        return self._run_training_telemetry_query(
            fields["model_ref"],
            "report-result",
            lambda: present_training_telemetry_report(
                report_query.execute(
                    GetTrainingTelemetryReportQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        model_ref=fields["model_ref"],
                        page_size=fields["page_size"],
                        cursor=fields["cursor"],
                    )
                )
            ),
        )

    def _gradient_interactions_response(
        self,
        owner: str,
        fields: TrainingTelemetryGradientRequestFields,
    ) -> bytes:
        gradient_query = self._get_gradient_interactions
        if gradient_query is None:
            raise RuntimeError("gradient interaction query is unavailable")
        return self._run_training_telemetry_query(
            fields["model_ref"],
            "gradient-interactions-result",
            lambda: present_gradient_interactions(
                gradient_query.execute(
                    GetGradientInteractionsQuery(
                        owner_subject=owner,
                        request_id=fields["request_id"],
                        model_ref=fields["model_ref"],
                        epoch=fields["epoch"],
                        page_size=fields["page_size"],
                        cursor=fields["cursor"],
                    )
                )
            ),
        )

    def _run_training_telemetry_query(
        self,
        model_ref: str,
        schema_name: str,
        query: Callable[[], JsonObject],
    ) -> bytes:
        try:
            result = query()
        except CatalogModelNotFound as exc:
            raise telemetry_model_not_found(exc.model_ref) from exc
        except InvalidTrainingTelemetryCursor as exc:
            raise invalid_telemetry_cursor() from exc
        except ExpiredTrainingTelemetryCursor as exc:
            raise expired_telemetry_cursor() from exc
        except InvalidatedTrainingTelemetryCursor as exc:
            raise invalidated_telemetry_cursor() from exc
        except TrainingTelemetrySnapshotCapacityExhausted as exc:
            raise telemetry_snapshot_capacity_exhausted(exc.operation) from exc
        except TrainingTelemetryStoredMetadataError as exc:
            raise telemetry_stored_metadata_invalid(model_ref, exc.path) from exc
        except TrainingTelemetryIntegrityError as exc:
            raise telemetry_integrity_failed(model_ref, exc.path) from exc
        except (
            TrainingTelemetryBackendUnavailable,
            ModelCatalogStoreUnavailable,
        ) as exc:
            raise telemetry_backend_unavailable(model_ref) from exc
        return training_telemetry_response(result, schema_name)

    def _target_head_diagnostics_report_response(
        self,
        owner: str,
        fields: TargetHeadDiagnosticsReportRequestFields,
    ) -> bytes:
        report_query = self._get_target_head_diagnostics_report
        if report_query is None:
            raise target_head_diagnostics_query_unavailable()
        try:
            result = report_query.execute(
                GetTargetHeadDiagnosticsReportQuery(
                    owner_subject=owner,
                    request_id=fields["request_id"],
                    model_ref=fields["model_ref"],
                    page_size=fields["page_size"],
                    cursor=fields["cursor"],
                )
            )
            return target_head_diagnostics_response(result)
        except CatalogModelNotFound as exc:
            raise target_head_diagnostics_model_not_found(exc.model_ref) from exc
        except TargetHeadDiagnosticsCursorError as exc:
            if exc.reason == "expired":
                raise expired_target_head_diagnostics_cursor() from exc
            if exc.reason == "invalidated":
                raise invalidated_target_head_diagnostics_cursor() from exc
            raise invalid_target_head_diagnostics_cursor() from exc
        except TrainingTelemetrySnapshotCapacityExhausted as exc:
            if exc.operation != "targetHeadDiagnostics":
                raise
            raise target_head_diagnostics_snapshot_capacity_exhausted() from exc
        except ModelCatalogStoreUnavailable as exc:
            raise target_head_diagnostics_backend_unavailable(
                fields["model_ref"]
            ) from exc
        except ServiceError as exc:
            if _stored_metadata_error(exc):
                raise target_head_diagnostics_stored_metadata_invalid(
                    fields["model_ref"],
                    _error_path(exc),
                ) from exc
            raise
        except ValueError as exc:
            raise target_head_diagnostics_integrity_failed(
                fields["model_ref"],
                "",
            ) from exc

    def capabilities(self, request_id: str) -> JsonObject:
        capabilities = self._service_status.capabilities()
        inventory = capabilities.device_inventory
        return response_document(
            request_id,
            limits=limits_to_api(capabilities.limits),
            devices={
                "cpu": {"available": True},
                "gpu": {
                    "available": inventory.cuda_capacity > 0,
                },
            },
            queries={
                "modelCatalog": True,
                "modelTopology": self._get_model_topology is not None,
                "trainingTelemetry": (
                    self._get_training_telemetry_report is not None
                ),
                "targetHeadDiagnostics": (
                    False
                    if self._get_target_head_diagnostics_report is None
                    else {
                        "contract": TARGET_HEAD_DIAGNOSTICS_CONTRACT_NAME,
                        "revision": TARGET_HEAD_DIAGNOSTICS_CONTRACT_REVISION,
                        "actions": [TARGET_HEAD_DIAGNOSTICS_REPORT_ACTION],
                        "artifactScope": "fullCommittedArtifact",
                        "encoderLayerDiagnosticsModes": [
                            "directComponentPerBatch",
                        ],
                        "encoderNormalizationOrders": list(
                            ENCODER_NORMALIZATION_ORDERS
                        ),
                        "maxEpochPageSize": TARGET_HEAD_DIAGNOSTICS_MAX_EPOCH_PAGE_SIZE,
                        "cursorTtlSeconds": TARGET_HEAD_DIAGNOSTICS_CURSOR_TTL_SECONDS,
                        "maxResponseBytes": TARGET_HEAD_DIAGNOSTICS_MAX_RESPONSE_BYTES,
                        "maxCommittedArtifactRows": MAX_COMMITTED_ARTIFACT_ROWS,
                        "maxRetainedSnapshotCount": TARGET_HEAD_DIAGNOSTICS_MAX_SNAPSHOT_COUNT,
                        "maxRetainedSnapshotBytes": TARGET_HEAD_DIAGNOSTICS_MAX_SNAPSHOT_BYTES,
                        "maxRetainedSnapshotTotalBytes": TARGET_HEAD_DIAGNOSTICS_MAX_SNAPSHOT_TOTAL_BYTES,
                        "availabilityStates": [
                            "notConfigured",
                            "pending",
                            "unavailable",
                            "available",
                        ],
                        "structuredErrors": True,
                    }
                ),
            },
            semantic=semantic_capabilities(),
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


def _stored_metadata_error(error: ServiceError) -> bool:
    return (
        error.code == ErrorCode.MODEL_CORRUPT
        and error.detail is not None
        and error.detail.get("reason") == "STORED_MODEL_METADATA_INVALID"
    )


def _error_path(error: ServiceError) -> str:
    if error.detail is None:
        return ""
    path = error.detail.get("path")
    return path if isinstance(path, str) else ""


def _create_command(
    owner: str,
    request: CreateRequestFields,
    document: JsonObject,
) -> CreateJobCommand:
    return CreateJobCommand(
        owner_subject=owner,
        request_id=request["request_id"],
        idempotency_key=request["idempotency_key"],
        request_hash=canonical_request_hash(document),
        job_id=request["job_id"],
        client_execution_id=request["client_execution_id"],
        operation=request["operation"],
        requested_device=device_from_api(request["device"]),
        prediction_column="prediction",
        source_encoding=dict(request["source_encoding"]),
        data_contract=dict(request["data_contract"]),
        model_contract=(
            None
            if request["model_contract"] is None
            else dict(request["model_contract"])
        ),
        semantic_digests=(
            None
            if request["semantic_digests"] is None
            else dict(request["semantic_digests"])
        ),
        model_label=request.get("model_label"),
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
        expected_logical_rows=request["expected_logical_rows"],
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
