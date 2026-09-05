from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import NotRequired, TypedDict, cast

from app.contracts.flight.v11.codec import (
    FlightContractError,
    FlightRequestSchema,
    validate_request_document,
)
from app.contracts.indexed_feature_blocks import canonical_source_encoding
from app.contracts.json_types import JsonObject
from app.contracts.semantic.v1 import ModelContract, SemanticContractError
from app.contracts.worker.v12.config import (
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v12.diagnostics import DiagnosticsConfig
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MAX_PAGE_ITEMS,
    MODEL_DESCRIBE_ACTION,
    OUTPUTS_LIST_ACTION,
    STATUS_ACTION,
)
from app.service.adapters.inbound.flight.errors import invalid
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode

_ACTION_SCHEMAS: dict[str, FlightRequestSchema] = {
    ACQUIRE_ACTION: "acquire",
    CANCEL_ACTION: "cancel",
    CAPABILITIES_ACTION: "query",
    CREATE_ACTION: "create",
    HEALTH_ACTION: "query",
    INPUT_CLOSE_ACTION: "input-close",
    INPUTS_LIST_ACTION: "inputs-list",
    MODEL_DESCRIBE_ACTION: "model-describe",
    OUTPUTS_LIST_ACTION: "outputs-list",
    STATUS_ACTION: "status",
}


class RequestIdFields(TypedDict):
    request_id: str


class StatusRequestFields(RequestIdFields):
    job_id: str


class DataContractFields(TypedDict):
    identity: str
    revision: int
    profile: str
    dataContractSha256: str
    seqLen: int
    featureDim: int


class CreateRequestFields(RequestIdFields):
    idempotency_key: str
    job_id: str
    client_execution_id: str
    operation: str
    device: str
    prediction_column: str
    source_encoding: JsonObject
    data_contract: DataContractFields
    model_contract: JsonObject
    semantic_digests: JsonObject
    model_label: NotRequired[str]
    model_ref: NotRequired[str]
    model_selector: NotRequired[str]
    model_config: NotRequired[ModelConfig]
    train_config: NotRequired[TrainConfig]
    initialization_kind: NotRequired[str]


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
    payload_count: int
    total_chunks: int
    total_rows: int
    total_native_rows: list[int]
    range_count: int
    total_bytes: int
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


class ModelDescribeRequestFields(RequestIdFields):
    model_ref: str
    model_selector: str


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
    | ModelDescribeRequestFields
)


def validate_action_request(
    action_name: str,
    document: JsonObject,
) -> ValidatedActionRequest:
    schema_name = _ACTION_SCHEMAS.get(action_name)
    if schema_name is None:
        raise invalid(f"unsupported action: {action_name}")
    _validate_schema(document, schema_name)
    request_id = _uuid(document, "requestId")

    if action_name in (CAPABILITIES_ACTION, HEALTH_ACTION):
        return {"request_id": request_id}
    if action_name == STATUS_ACTION:
        return {"request_id": request_id, "job_id": _uuid(document, "jobId")}
    if action_name == CREATE_ACTION:
        return _validate_create(document, request_id)
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
    if action_name == MODEL_DESCRIBE_ACTION:
        return _validate_model_describe(document, request_id)
    raise invalid(f"unsupported action: {action_name}")


def validate_upload_metadata(document: JsonObject) -> UploadMetadataFields:
    _validate_schema(document, "upload-metadata")
    schema_id = _string(document, "schemaId")
    return {
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "fencing_token": _fencing_token(document, "fencingToken"),
        "payload_id": _uuid(document, "payloadId"),
        "ordinal": _integer(document, "ordinal"),
        "schema_id": schema_id,
        "input_kind": "fit" if schema_id == FIT_SCHEMA_ID else "predict",
        "data_contract_sha256": _string(document, "dataContractSha256"),
        "chunks": _integer(document, "chunks"),
        "rows": _integer(document, "logicalRows"),
        "native_rows": tuple(_integer_list(document, "nativeRows")),
    }


def _validate_schema(
    document: JsonObject,
    schema_name: FlightRequestSchema,
) -> None:
    try:
        validate_request_document(document, schema_name)
    except FlightContractError as exc:
        raise invalid(_schema_error_message(document, schema_name, exc)) from exc


def _schema_error_message(
    document: Mapping[str, object],
    schema_name: FlightRequestSchema,
    error: FlightContractError,
) -> str:
    validation_error = error.validation_error
    if validation_error is None:
        return str(error)
    if validation_error.validator == "oneOf":
        if schema_name == "create":
            return "create request must match exactly one operation form"
        if schema_name == "inputs-list":
            return "snapshotRevision and cursor must be supplied together"
        if schema_name == "model-describe":
            return "model.describe requires exactly one of modelRef or modelAlias"
    if (
        schema_name == "upload-metadata"
        and validation_error.validator == "additionalProperties"
        and "requestId" in document
    ):
        return "DoPut metadata must not contain requestId"
    return str(error)


def _validate_create(
    document: JsonObject,
    request_id: str,
) -> CreateRequestFields:
    operation = _string(document, "operation")
    data_contract = _data_contract(_object(document, "dataContract"))
    try:
        model_contract = ModelContract.from_document(
            document["modelContract"]
        )
    except SemanticContractError as exc:
        code = (
            ErrorCode.FAILED_PRECONDITION
            if exc.reason in {
                "LANGUAGE_REVISION_UNAVAILABLE",
                "PRIMITIVE_UNAVAILABLE",
            }
            else ErrorCode.INVALID_ARGUMENT
        )
        raise ServiceError(
            code,
            str(exc),
            detail={
                "code": code.value,
                "reason": exc.reason,
                "path": "/modelContract" + exc.path,
                "message": str(exc),
            },
        ) from exc
    model_config = ModelConfig.from_manifest(model_contract.model_config)
    if (
        model_config.seq_len != data_contract["seqLen"]
        or model_config.feature_dim != data_contract["featureDim"]
    ):
        raise invalid(
            "modelContract.modelConfig geometry must match dataContract"
        )
    try:
        source_encoding = canonical_source_encoding(
            document["sourceEncoding"],
            feature_dim=data_contract["featureDim"],
        )
    except ValueError as exc:
        raise invalid(str(exc)) from exc
    common: CreateRequestFields = {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "operation": operation,
        "device": _string(document, "device"),
        "prediction_column": cast(str, document.get("predictionColumn", "out")),
        "source_encoding": source_encoding,
        "data_contract": data_contract,
        "model_contract": model_contract.to_document(),
        "semantic_digests": model_contract.digests(
            data_contract["dataContractSha256"]
        ),
    }
    if operation == "fit":
        train_config = _train_config(
            cast(Mapping[str, object], document.get("trainingConfig", {}))
        )
        diagnostics = _diagnostics_config(document.get("diagnostics"))
        train_config = replace(train_config, diagnostics=diagnostics)
        common["model_label"] = _string(document, "modelLabel")
        common["model_config"] = model_config
        common["train_config"] = train_config
        initialization = _object(document, "initialization")
        initialization_kind = _string(initialization, "kind")
        common["initialization_kind"] = initialization_kind
        if initialization_kind == "publishedModel":
            common["model_ref"] = _string(initialization, "modelRef")
            common["model_selector"] = "modelRef"
    else:
        selector = "modelRef" if "modelRef" in document else "modelAlias"
        common["model_ref"] = _string(document, selector)
        common["model_selector"] = selector
    return common


def _validate_acquire(
    document: JsonObject,
    request_id: str,
) -> AcquireRequestFields:
    previous = _uuid(document, "previousClientExecutionId")
    current = _uuid(document, "clientExecutionId")
    if previous == current:
        raise invalid("clientExecutionId must change during acquire")
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "previous_client_execution_id": previous,
        "expected_fencing_token": _fencing_token(
            document,
            "expectedFencingToken",
        ),
        "client_execution_id": current,
    }


def _validate_inputs_list(
    document: JsonObject,
    request_id: str,
) -> InputsListRequestFields:
    after_revision = _integer(document, "afterRevision")
    snapshot = _optional_integer(document, "snapshotRevision")
    cursor = _optional_integer(document, "cursor")
    if (
        snapshot is not None
        and cursor is not None
        and not after_revision <= cursor <= snapshot
    ):
        raise invalid(
            "pagination must satisfy afterRevision <= cursor <= snapshotRevision"
        )
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
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "fencing_token": _fencing_token(document, "fencingToken"),
        "payload_count": _integer(document, "payloadCount"),
        "total_chunks": _integer(document, "totalChunks"),
        "total_rows": _integer(document, "totalLogicalRows"),
        "total_native_rows": _integer_list(document, "totalNativeRows"),
        "range_count": _integer(document, "rangeCount"),
        "total_bytes": _integer(document, "totalBytes"),
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


def _validate_cancel(
    document: JsonObject,
    request_id: str,
) -> CancelRequestFields:
    return {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "fencing_token": _fencing_token(document, "fencingToken"),
    }


def _validate_model_describe(
    document: JsonObject,
    request_id: str,
) -> ModelDescribeRequestFields:
    selector = "modelRef" if "modelRef" in document else "modelAlias"
    return {
        "request_id": request_id,
        "model_ref": _string(document, selector),
        "model_selector": selector,
    }


def _data_contract(document: Mapping[str, object]) -> DataContractFields:
    return {
        "identity": _string(document, "identity"),
        "revision": _integer(document, "revision"),
        "profile": _string(document, "profile"),
        "dataContractSha256": _string(document, "dataContractSha256"),
        "seqLen": _integer(document, "seqLen"),
        "featureDim": _integer(document, "featureDim"),
    }


def _diagnostics_config(document: object) -> DiagnosticsConfig:
    if document is None:
        return DiagnosticsConfig()
    try:
        return DiagnosticsConfig.from_document(document)
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid diagnostics: {exc}") from exc


def _train_config(document: Mapping[str, object]) -> TrainConfig:
    names = {
        "lr": "lr",
        "batchSize": "batch_size",
        "epochs": "epochs",
        "useAmp": "use_amp",
        "weightDecay": "weight_decay",
        "selection": "selection",
        "seed": "seed",
        "deterministic": "deterministic",
    }
    mapped = {names[key]: value for key, value in document.items()}
    try:
        if not mapped:
            return TrainConfig()
        config = TrainConfig.from_dict(mapped)
        if config is None:
            raise ValueError("training configuration must not be empty")
        return config
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid trainingConfig: {exc}") from exc


def _uuid(document: Mapping[str, object], key: str) -> str:
    return _string(document, key).lower()


def _fencing_token(document: Mapping[str, object], key: str) -> int:
    parsed = int(_string(document, key))
    if parsed > 2**63 - 1:
        raise invalid(f"{key} exceeds the supported range")
    return parsed


def _object(document: Mapping[str, object], key: str) -> Mapping[str, object]:
    return cast(Mapping[str, object], document[key])


def _string(document: Mapping[str, object], key: str) -> str:
    return cast(str, document[key])


def _integer(document: Mapping[str, object], key: str) -> int:
    return cast(int, document[key])


def _optional_integer(document: Mapping[str, object], key: str) -> int | None:
    if key not in document:
        return None
    return _integer(document, key)


def _integer_list(document: Mapping[str, object], key: str) -> list[int]:
    return cast(list[int], list(cast(list[object], document[key])))


def _page_limit(document: Mapping[str, object]) -> int:
    if "limit" not in document:
        return MAX_PAGE_ITEMS
    return _integer(document, "limit")


__all__ = [
    "validate_action_request",
    "validate_upload_metadata",
]
