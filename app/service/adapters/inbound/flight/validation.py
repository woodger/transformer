from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import NotRequired, TypedDict, cast

from app.contracts.flight.v19.codec import (
    FlightContractError,
    FlightRequestSchema,
    validate_request_document,
)
from app.contracts.flight.v19.source_encoding import canonical_source_encoding
from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v5 import (
    ModelCatalogContractError,
    validate_catalog_document,
)
from app.contracts.model_topology.v2 import (
    ModelTopologyContractError,
    validate_model_topology_document,
)
from app.contracts.semantic.v4 import ModelContract, SemanticContractError
from app.contracts.target_head_diagnostics.v2 import (
    TargetHeadDiagnosticsContractError,
    validate_target_head_diagnostics_document,
)
from app.contracts.training_telemetry.v4 import (
    TrainingTelemetryContractError,
    validate_training_telemetry_document,
)
from app.contracts.worker.v17.config import ModelConfig, TrainConfig
from app.contracts.worker.v17.diagnostics import DiagnosticsConfig
from app.contracts.worker.v17.model_definition import (
    resolved_semantic_digests,
)
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    FIT_CREATE_ACTION,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MAX_PAGE_ITEMS,
    MODEL_TOPOLOGY_DETAIL_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_CREATE_ACTION,
    STATUS_ACTION,
)
from app.service.adapters.inbound.flight.errors import invalid
from app.service.adapters.inbound.flight.model_catalog import (
    invalid_catalog_cursor,
    invalid_catalog_query,
)
from app.service.adapters.inbound.flight.model_topology import (
    invalid_model_topology_query,
)
from app.service.adapters.inbound.flight.mutation_lease import (
    MutationLeaseError,
    decode_mutation_lease,
)
from app.service.adapters.inbound.flight.target_head_diagnostics import (
    invalid_target_head_diagnostics_cursor,
    invalid_target_head_diagnostics_query,
)
from app.service.adapters.inbound.flight.training_telemetry import (
    invalid_telemetry_cursor,
    invalid_telemetry_query,
)
from app.service.domain.errors import ServiceError
from app.service.domain.initialization import validate_requested_initialization
from app.service.domain.job import ErrorCode

_ACTION_SCHEMAS: dict[str, FlightRequestSchema] = {
    ACQUIRE_ACTION: "acquire",
    CANCEL_ACTION: "cancel",
    CAPABILITIES_ACTION: "query",
    FIT_CREATE_ACTION: "fit-create",
    HEALTH_ACTION: "query",
    INPUT_CLOSE_ACTION: "input-close",
    INPUTS_LIST_ACTION: "inputs-list",
    OUTPUTS_LIST_ACTION: "outputs-list",
    PREDICT_CREATE_ACTION: "predict-create",
    STATUS_ACTION: "status",
}


class RequestIdFields(TypedDict):
    request_id: str


class StatusRequestFields(RequestIdFields):
    job_id: str


class DataBindingFields(TypedDict):
    dataContractSha256: str
    seqLen: int
    featureDim: int
    inputLayout: JsonObject


class CreateRequestFields(RequestIdFields):
    idempotency_key: str
    job_id: str
    client_execution_id: str
    operation: str
    device: str
    source_encoding: JsonObject
    data_contract: JsonObject
    model_contract: JsonObject | None
    semantic_digests: JsonObject | None
    model_label: NotRequired[str]
    model_ref: NotRequired[str]
    model_config: NotRequired[ModelConfig]
    train_config: NotRequired[TrainConfig]
    initialization_source: NotRequired[str]
    requested_initialization: NotRequired[JsonObject]


class AcquireRequestFields(RequestIdFields):
    idempotency_key: str
    job_id: str
    previous_client_execution_id: str
    expected_fencing_token: int
    client_execution_id: str


class InputsListRequestFields(RequestIdFields):
    job_id: str
    after_revision: int
    snapshot_revision: int | None
    cursor: int | None
    limit: int


class InputCloseRequestFields(RequestIdFields):
    idempotency_key: str
    job_id: str
    client_execution_id: str
    fencing_token: int
    expected_logical_rows: int | None
    manifest_sha256: str


class OutputsListRequestFields(RequestIdFields):
    job_id: str
    cursor: int | None
    limit: int


class CancelRequestFields(RequestIdFields):
    idempotency_key: str
    job_id: str
    client_execution_id: str
    fencing_token: int


class ModelCatalogListRequestFields(RequestIdFields):
    page_size: int
    cursor: str | None


class ModelCatalogDetailRequestFields(RequestIdFields):
    model_ref: str


class ModelTopologyDetailRequestFields(RequestIdFields):
    model_ref: str


class TrainingTelemetryReportRequestFields(RequestIdFields):
    model_ref: str
    page_size: int
    cursor: str | None


class TrainingTelemetryGradientRequestFields(RequestIdFields):
    model_ref: str
    epoch: int
    page_size: int
    cursor: str | None


class TargetHeadDiagnosticsReportRequestFields(RequestIdFields):
    model_ref: str
    page_size: int
    cursor: str | None


class UploadMetadataFields(TypedDict):
    job_id: str
    client_execution_id: str
    fencing_token: int
    payload_id: str
    ordinal: int
    schema_id: str
    input_kind: str
    data_contract_sha256: str
    chunks: int
    rows: int
    native_rows: tuple[int, ...]


ValidatedActionRequest = (
    RequestIdFields
    | StatusRequestFields
    | CreateRequestFields
    | AcquireRequestFields
    | InputsListRequestFields
    | InputCloseRequestFields
    | OutputsListRequestFields
    | CancelRequestFields
    | ModelCatalogListRequestFields
    | ModelCatalogDetailRequestFields
    | ModelTopologyDetailRequestFields
    | TrainingTelemetryReportRequestFields
    | TrainingTelemetryGradientRequestFields
    | TargetHeadDiagnosticsReportRequestFields
)


def validate_action_request(
    action_name: str,
    document: JsonObject,
) -> ValidatedActionRequest:
    from app.contracts.model_catalog.v5.constants import (
        DETAIL_ACTION as MODEL_CATALOG_DETAIL_ACTION,
        LIST_ACTION as MODEL_CATALOG_LIST_ACTION,
    )
    from app.contracts.target_head_diagnostics.v2.constants import (
        REPORT_ACTION as TARGET_HEAD_DIAGNOSTICS_REPORT_ACTION,
    )
    from app.contracts.training_telemetry.v4.constants import (
        GRADIENT_INTERACTIONS_ACTION as TRAINING_TELEMETRY_GRADIENT_ACTION,
        REPORT_ACTION as TRAINING_TELEMETRY_REPORT_ACTION,
    )

    if action_name == MODEL_CATALOG_LIST_ACTION:
        return _validate_model_catalog_list(document)
    if action_name == MODEL_CATALOG_DETAIL_ACTION:
        return _validate_model_catalog_detail(document)
    if action_name == MODEL_TOPOLOGY_DETAIL_ACTION:
        return _validate_model_topology_detail(document)
    if action_name == TRAINING_TELEMETRY_REPORT_ACTION:
        return _validate_training_telemetry_report(document)
    if action_name == TRAINING_TELEMETRY_GRADIENT_ACTION:
        return _validate_training_telemetry_gradient(document)
    if action_name == TARGET_HEAD_DIAGNOSTICS_REPORT_ACTION:
        return _validate_target_head_diagnostics_report(document)
    schema_name = _ACTION_SCHEMAS.get(action_name)
    if schema_name is None:
        raise invalid(f"unsupported action: {action_name}")
    _validate_schema(document, schema_name)
    request_id = _uuid(document, "requestId")

    if action_name in (CAPABILITIES_ACTION, HEALTH_ACTION):
        return {"request_id": request_id}
    if action_name == STATUS_ACTION:
        return {"request_id": request_id, "job_id": _uuid(document, "jobId")}
    if action_name == FIT_CREATE_ACTION:
        return _validate_fit_create(document, request_id)
    if action_name == PREDICT_CREATE_ACTION:
        return _validate_predict_create(document, request_id)
    if action_name == ACQUIRE_ACTION:
        return _validate_acquire(document, request_id)
    if action_name == INPUTS_LIST_ACTION:
        return _validate_inputs_list(document, request_id)
    if action_name == INPUT_CLOSE_ACTION:
        return _validate_input_close(document, request_id)
    if action_name == OUTPUTS_LIST_ACTION:
        return _validate_outputs_list(document, request_id)
    if action_name == CANCEL_ACTION:
        return _validate_cancel(document, request_id)
    raise invalid(f"unsupported action: {action_name}")


def validate_upload_metadata(document: JsonObject) -> UploadMetadataFields:
    _validate_schema(document, "upload-metadata")
    client_execution_id, fencing_token = _lease(document)
    return {
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": client_execution_id,
        "fencing_token": fencing_token,
        "payload_id": _uuid(document, "payloadId"),
        "ordinal": _integer(document, "ordinal"),
        # Эти значения разрешаются из задания в области владельца, а не
        # принимаются от транспортного клиента.
        "schema_id": "",
        "input_kind": "",
        "data_contract_sha256": "",
        "chunks": _integer(document, "chunks"),
        "rows": _integer(document, "logicalRows"),
        "native_rows": tuple(_integer_list(document, "nativeRows")),
    }


def _validate_schema(document: JsonObject, schema_name: FlightRequestSchema) -> None:
    try:
        validate_request_document(document, schema_name)
    except FlightContractError as exc:
        validation_error = exc.validation_error
        if (
            schema_name == "fit-create"
            and validation_error is not None
            and tuple(validation_error.absolute_path)[:1] == ("modelContract",)
        ):
            try:
                ModelContract.from_document(document.get("modelContract"))
            except SemanticContractError as semantic_error:
                raise _semantic_service_error(semantic_error) from exc
        raise invalid(_schema_error_message(schema_name, exc)) from exc


def _schema_error_message(schema_name: str, error: FlightContractError) -> str:
    validation_error = error.validation_error
    if validation_error is not None and validation_error.validator == "oneOf":
        if schema_name == "inputs-list":
            return "snapshotRevision and cursor must be supplied together"
    return str(error)


def _validate_fit_create(
    document: JsonObject,
    request_id: str,
) -> CreateRequestFields:
    data_binding = _data_binding(_object(document, "dataBinding"))
    try:
        model_contract = ModelContract.from_document(document["modelContract"])
        model_config = ModelConfig.from_tuning(
            model_contract.model_tuning,
            seq_len=data_binding["seqLen"],
            feature_dim=data_binding["featureDim"],
        )
    except SemanticContractError as exc:
        raise _semantic_service_error(exc) from exc
    except ValueError as exc:
        raise invalid(f"invalid modelTuning: {exc}") from exc
    train_config = _train_config(_object(document, "trainingConfig"))
    diagnostics = _diagnostics_config(document["diagnostics"])
    train_config = replace(train_config, diagnostics=diagnostics)
    try:
        initialization = validate_requested_initialization(
            _object(document, "initialization")
        )
    except ValueError as exc:
        raise invalid(f"invalid initialization: {exc}") from exc
    common: CreateRequestFields = {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": str(uuid.uuid4()),
        "client_execution_id": str(uuid.uuid4()),
        "operation": "fit",
        "device": _string(document, "device"),
        "source_encoding": data_binding["inputLayout"],
        "data_contract": _internal_data_contract(data_binding),
        "model_contract": model_contract.to_document(),
        "semantic_digests": resolved_semantic_digests(
            model_contract,
            data_binding["dataContractSha256"],
            model_config,
        ),
        "model_label": _string(document, "modelLabel"),
        "model_config": model_config,
        "train_config": train_config,
        "initialization_source": _string(initialization, "source"),
        "requested_initialization": initialization,
    }
    if initialization["source"] == "publishedModel":
        common["model_ref"] = _string(initialization, "modelRef")
    return common


def _validate_predict_create(
    document: JsonObject,
    request_id: str,
) -> CreateRequestFields:
    data_binding = _data_binding(_object(document, "dataBinding"))
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": str(uuid.uuid4()),
        "client_execution_id": str(uuid.uuid4()),
        "operation": "predict",
        "device": _string(document, "device"),
        "source_encoding": data_binding["inputLayout"],
        "data_contract": _internal_data_contract(data_binding),
        "model_contract": None,
        "semantic_digests": None,
        "model_ref": _string(document, "modelRef"),
    }


def _validate_acquire(document: JsonObject, request_id: str) -> AcquireRequestFields:
    previous, fence = _lease(document)
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "previous_client_execution_id": previous,
        "expected_fencing_token": fence,
        "client_execution_id": str(uuid.uuid4()),
    }


def _validate_inputs_list(
    document: JsonObject,
    request_id: str,
) -> InputsListRequestFields:
    after_revision = _integer(document, "afterRevision")
    snapshot = _optional_integer(document, "snapshotRevision")
    cursor = _optional_integer(document, "cursor")
    if snapshot is not None and cursor is not None and not after_revision <= cursor <= snapshot:
        raise invalid("pagination must satisfy afterRevision <= cursor <= snapshotRevision")
    return {
        "request_id": request_id,
        "job_id": _uuid(document, "jobId"),
        "after_revision": after_revision,
        "snapshot_revision": snapshot,
        "cursor": cursor,
        "limit": _page_limit(document),
    }


def _validate_input_close(
    document: JsonObject,
    request_id: str,
) -> InputCloseRequestFields:
    client_execution_id, fencing_token = _lease(document)
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": client_execution_id,
        "fencing_token": fencing_token,
        "expected_logical_rows": _optional_integer(document, "expectedLogicalRows"),
        "manifest_sha256": _string(document, "manifestSha256"),
    }


def _validate_outputs_list(
    document: JsonObject,
    request_id: str,
) -> OutputsListRequestFields:
    return {
        "request_id": request_id,
        "job_id": _uuid(document, "jobId"),
        "cursor": _optional_integer(document, "cursor"),
        "limit": _page_limit(document),
    }


def _validate_cancel(document: JsonObject, request_id: str) -> CancelRequestFields:
    client_execution_id, fencing_token = _lease(document)
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": client_execution_id,
        "fencing_token": fencing_token,
    }


def _validate_model_catalog_list(document: JsonObject) -> ModelCatalogListRequestFields:
    _validate_catalog_schema(document, "list-request")
    return {
        "request_id": _uuid(document, "requestId"),
        "page_size": _integer(document, "pageSize"),
        "cursor": cast(str | None, document["cursor"]),
    }


def _validate_model_catalog_detail(document: JsonObject) -> ModelCatalogDetailRequestFields:
    _validate_catalog_schema(document, "detail-request")
    return {
        "request_id": _uuid(document, "requestId"),
        "model_ref": _string(document, "modelRef"),
    }


def _validate_model_topology_detail(
    document: JsonObject,
) -> ModelTopologyDetailRequestFields:
    _validate_model_topology_schema(document, "detail-request")
    return {
        "request_id": _uuid(document, "requestId"),
        "model_ref": _string(document, "modelRef"),
    }


def _validate_training_telemetry_report(document: JsonObject) -> TrainingTelemetryReportRequestFields:
    _validate_training_telemetry_schema(document, "report-request")
    return {
        "request_id": _uuid(document, "requestId"),
        "model_ref": _string(document, "modelRef"),
        "page_size": _integer(document, "pageSize"),
        "cursor": cast(str | None, document["cursor"]),
    }


def _validate_training_telemetry_gradient(document: JsonObject) -> TrainingTelemetryGradientRequestFields:
    _validate_training_telemetry_schema(document, "gradient-interactions-request")
    return {
        "request_id": _uuid(document, "requestId"),
        "model_ref": _string(document, "modelRef"),
        "epoch": _integer(document, "epoch"),
        "page_size": _integer(document, "pageSize"),
        "cursor": cast(str | None, document["cursor"]),
    }


def _validate_target_head_diagnostics_report(
    document: JsonObject,
) -> TargetHeadDiagnosticsReportRequestFields:
    try:
        validate_target_head_diagnostics_document(document, "report-request")
    except TargetHeadDiagnosticsContractError as exc:
        path = _contract_error_path(exc)
        if path == "/cursor" and document.get("cursor") is not None:
            raise invalid_target_head_diagnostics_cursor() from exc
        raise invalid_target_head_diagnostics_query(str(exc), path) from exc
    return {
        "request_id": _uuid(document, "requestId"),
        "model_ref": _string(document, "modelRef"),
        "page_size": _integer(document, "pageSize"),
        "cursor": cast(str | None, document["cursor"]),
    }


def _validate_catalog_schema(document: JsonObject, schema_name: str) -> None:
    try:
        validate_catalog_document(document, schema_name)
    except ModelCatalogContractError as exc:
        path = _contract_error_path(exc)
        if schema_name == "list-request" and path == "/cursor" and document.get("cursor") is not None:
            raise invalid_catalog_cursor() from exc
        raise invalid_catalog_query(str(exc), path) from exc


def _validate_training_telemetry_schema(document: JsonObject, schema_name: str) -> None:
    try:
        validate_training_telemetry_document(document, schema_name)
    except TrainingTelemetryContractError as exc:
        path = _contract_error_path(exc)
        if path == "/cursor" and document.get("cursor") is not None:
            raise invalid_telemetry_cursor() from exc
        raise invalid_telemetry_query(str(exc), path) from exc


def _validate_model_topology_schema(document: JsonObject, schema_name: str) -> None:
    try:
        validate_model_topology_document(document, schema_name)
    except ModelTopologyContractError as exc:
        raise invalid_model_topology_query(
            str(exc),
            _contract_error_path(exc),
        ) from exc


def _data_binding(document: Mapping[str, object]) -> DataBindingFields:
    geometry = _object(document, "tensorGeometry")
    feature_dim = _integer(geometry, "featureDim")
    layout = canonical_source_encoding(
        document["inputLayout"],
        feature_dim=feature_dim,
    )
    return {
        "dataContractSha256": _string(document, "dataContractSha256"),
        "seqLen": _integer(geometry, "seqLen"),
        "featureDim": feature_dim,
        "inputLayout": layout,
    }


def _internal_data_contract(data_binding: DataBindingFields) -> JsonObject:
    return {
        "dataContractSha256": data_binding["dataContractSha256"],
        "seqLen": data_binding["seqLen"],
        "featureDim": data_binding["featureDim"],
    }


def _diagnostics_config(document: object) -> DiagnosticsConfig:
    try:
        return DiagnosticsConfig.from_document({
            "schemaVersion": 2,
            **_mapping(document),
        })
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid diagnostics: {exc}") from exc


def _train_config(document: Mapping[str, object]) -> TrainConfig:
    names = {
        "learningRate": "lr",
        "batchSize": "batch_size",
        "epochs": "epochs",
        "mixedPrecision": "use_amp",
        "weightDecay": "weight_decay",
        "selection": "selection",
        "seed": "seed",
        "deterministic": "deterministic",
    }
    try:
        config = TrainConfig.from_dict({names[key]: value for key, value in document.items()})
        if config is None:
            raise ValueError("training configuration must not be empty")
        return config
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid trainingConfig: {exc}") from exc


def _lease(document: Mapping[str, object]) -> tuple[str, int]:
    try:
        execution, fence = decode_mutation_lease(_string(document, "mutationLease"))
        parsed = uuid.UUID(execution)
    except (MutationLeaseError, ValueError) as exc:
        raise invalid("mutationLease is invalid") from exc
    return str(parsed), fence


def _uuid(document: Mapping[str, object], key: str) -> str:
    try:
        return str(uuid.UUID(_string(document, key)))
    except ValueError as exc:
        raise invalid(f"{key} must be a UUID") from exc


def _object(document: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = document[key]
    if not isinstance(value, Mapping):
        raise invalid(f"{key} must be an object")
    return cast(Mapping[str, object], value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return cast(Mapping[str, object], value)


def _string(document: Mapping[str, object], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise invalid(f"{key} must be a string")
    return value


def _integer(document: Mapping[str, object], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid(f"{key} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise invalid(f"{key} must be an integer")
    return int(value)


def _optional_integer(document: Mapping[str, object], key: str) -> int | None:
    if key not in document:
        return None
    return _integer(document, key)


def _integer_list(document: Mapping[str, object], key: str) -> list[int]:
    values = document[key]
    if not isinstance(values, list):
        raise invalid(f"{key} must be an array")
    parsed_values = cast(list[object], values)
    return [_integer(cast(Mapping[str, object], {key: value}), key) for value in parsed_values]


def _page_limit(document: Mapping[str, object]) -> int:
    return _integer(document, "limit") if "limit" in document else MAX_PAGE_ITEMS


def _contract_error_path(error: object) -> str:
    validation_error = getattr(error, "validation_error", None)
    if validation_error is None:
        return ""
    path = "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1")
        for part in validation_error.absolute_path
    )
    return "" if path == "/" else path


def _semantic_service_error(error: SemanticContractError) -> ServiceError:
    code = ErrorCode.INVALID_ARGUMENT
    return ServiceError(
        code,
        str(error),
        detail={
            "code": code.value,
            "reason": error.reason,
            "path": error.path,
            "message": str(error),
        },
    )


__all__ = [
    "AcquireRequestFields",
    "CancelRequestFields",
    "CreateRequestFields",
    "InputCloseRequestFields",
    "InputsListRequestFields",
    "ModelCatalogDetailRequestFields",
    "ModelCatalogListRequestFields",
    "ModelTopologyDetailRequestFields",
    "OutputsListRequestFields",
    "RequestIdFields",
    "StatusRequestFields",
    "TargetHeadDiagnosticsReportRequestFields",
    "TrainingTelemetryGradientRequestFields",
    "TrainingTelemetryReportRequestFields",
    "UploadMetadataFields",
    "ValidatedActionRequest",
    "validate_action_request",
    "validate_upload_metadata",
]
