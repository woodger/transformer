from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import NotRequired, TypedDict, cast

from app.contracts.flight.v4.codec import (
    FlightContractError,
    FlightRequestSchema,
    validate_request_document,
)
from app.contracts.json_types import JsonObject
from app.contracts.worker.v6.config import (
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v6.objective import ml_contract
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
    id: str
    version: int
    data_contract_sha256: str
    seq_len: int
    feature_dim: int
    target_schema_id: str


class MlContractFields(TypedDict):
    targetSchemaId: str
    predictionSchemaId: str
    objectiveId: str
    objectiveConfigSha256: str
    checkpointFormat: str
    targetWidth: int
    predictionSpace: str


class CreateRequestFields(RequestIdFields):
    idempotency_key: str
    job_id: str
    client_execution_id: str
    operation: str
    device: str
    prediction_column: str
    data_contract: DataContractFields
    ml_contract: MlContractFields
    model_label: NotRequired[str]
    model_ref: NotRequired[str]
    model_selector: NotRequired[str]
    model_config: NotRequired[ModelConfig]
    train_config: NotRequired[TrainConfig]


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
    total_rows: int
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
    rows: int


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
        "rows": _integer(document, "rows"),
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
    requested_ml_contract = _ml_contract(_object(document, "mlContract"))
    common: CreateRequestFields = {
        "request_id": request_id,
        "idempotency_key": _string(document, "idempotencyKey"),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "operation": operation,
        "device": _string(document, "device"),
        "prediction_column": cast(str, document.get("predictionColumn", "out")),
        "data_contract": data_contract,
        "ml_contract": requested_ml_contract,
    }
    if operation == "fit":
        model_config = _model_config(_object(document, "modelConfig"))
        if model_config.seq_len != data_contract["seq_len"]:
            raise invalid("modelConfig.seqLen must match dataContract.seqLen")
        resolved_model_config = replace(
            model_config,
            feature_dim=data_contract["feature_dim"],
        )
        train_config = _train_config(
            cast(Mapping[str, object], document.get("trainingConfig", {}))
        )
        if requested_ml_contract != ml_contract(train_config):
            raise invalid(
                "mlContract does not match the target-aligned training objective"
            )
        common["model_label"] = _string(document, "modelLabel")
        common["model_config"] = resolved_model_config
        common["train_config"] = train_config
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
        "total_rows": _integer(document, "totalRows"),
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
        "id": _string(document, "id"),
        "version": _integer(document, "version"),
        "data_contract_sha256": _string(document, "dataContractSha256"),
        "seq_len": _integer(document, "seqLen"),
        "feature_dim": _integer(document, "featureDim"),
        "target_schema_id": _string(document, "targetSchemaId"),
    }


def _ml_contract(document: Mapping[str, object]) -> MlContractFields:
    return {
        "targetSchemaId": _string(document, "targetSchemaId"),
        "predictionSchemaId": _string(document, "predictionSchemaId"),
        "objectiveId": _string(document, "objectiveId"),
        "objectiveConfigSha256": _string(document, "objectiveConfigSha256"),
        "checkpointFormat": _string(document, "checkpointFormat"),
        "targetWidth": _integer(document, "targetWidth"),
        "predictionSpace": _string(document, "predictionSpace"),
    }


def _model_config(document: Mapping[str, object]) -> ModelConfig:
    mapped: dict[str, object] = {
        "seq_len": document["seqLen"],
        "hidden": document.get(
            "hidden",
            ModelConfig.__dataclass_fields__["hidden"].default,
        ),
        "layers": document.get(
            "layers",
            ModelConfig.__dataclass_fields__["layers"].default,
        ),
        "dropout": document.get(
            "dropout",
            ModelConfig.__dataclass_fields__["dropout"].default,
        ),
        "nhead": document.get(
            "nhead",
            ModelConfig.__dataclass_fields__["nhead"].default,
        ),
        "context_mode": document.get(
            "mode",
            ModelConfig.__dataclass_fields__["context_mode"].default,
        ),
    }
    try:
        config = ModelConfig.from_dict(mapped)
        if config is None:
            raise ValueError("model configuration must not be empty")
        return config
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid modelConfig: {exc}") from exc


def _train_config(document: Mapping[str, object]) -> TrainConfig:
    names = {
        "lr": "lr",
        "batchSize": "batch_size",
        "epochs": "epochs",
        "lossStage": "loss_stage",
        "lossSchedule": "loss_schedule",
        "stageSize": "stage_size",
        "useAmp": "use_amp",
        "weightDecay": "weight_decay",
        "directLossWeights": "direct_loss_weights",
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


def _page_limit(document: Mapping[str, object]) -> int:
    if "limit" not in document:
        return MAX_PAGE_ITEMS
    return _integer(document, "limit")


__all__ = [
    "validate_action_request",
    "validate_upload_metadata",
]
