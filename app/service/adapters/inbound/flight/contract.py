from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import (
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    overload,
    runtime_checkable,
)

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v3.config import (
    ModelConfig,
    TrainConfig,
    train_config_to_manifest,
)
from app.contracts.worker.v3.objective import (
    CHECKPOINT_FORMAT,
    OBJECTIVE_ID,
    PREDICTION_SCHEMA_ID,
    TARGET_SCHEMA_ID,
    TARGET_WIDTH,
    ml_contract,
)
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CONTRACT_PATH_VERSION,
    CONTRACT_VERSION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MAX_PAGE_ITEMS,
    MAX_PAYLOADS_PER_JOB,
    MODEL_DESCRIBE_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_SCHEMA_ID,
    STATUS_ACTION,
    SUPPORTED_DEVICES,
    SUPPORTED_OPERATIONS,
)
from app.service.adapters.inbound.flight.errors import invalid
from app.service.domain.input_manifest import canonical_receipts, manifest_sha256

MAX_ACTION_DOCUMENT_BYTES = 64 * 1024
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PREDICTION_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FENCING_TOKEN = re.compile(r"^[1-9][0-9]{0,18}$")

_COMMON = {"contract", "version", "requestId"}
_MUTATING = _COMMON | {"idempotencyKey"}
_FENCED = _MUTATING | {"jobId", "clientExecutionId", "fencingToken"}


@runtime_checkable
class _ArrowBuffer(Protocol):
    def to_pybytes(self) -> bytes: ...


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


def parse_action_body(body: object) -> JsonObject:
    if isinstance(body, _ArrowBuffer):
        body = body.to_pybytes()
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise invalid("action body must be UTF-8 JSON bytes")
    if isinstance(body, bytes):
        body_bytes = body
    elif isinstance(body, bytearray):
        body_bytes = bytes(body)
    else:
        body_bytes = body.tobytes()
    if len(body_bytes) > MAX_ACTION_DOCUMENT_BYTES:
        raise invalid(
            f"action body exceeds {MAX_ACTION_DOCUMENT_BYTES} bytes"
        )
    try:
        document = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise invalid("action body must be a valid UTF-8 JSON document") from exc
    if not isinstance(document, dict):
        raise invalid("action body must be a JSON object")
    return cast(JsonObject, document)


def validate_action_request(
    action_name: str,
    document: JsonObject,
) -> ValidatedActionRequest:
    request_id = _validate_common(document)
    if action_name in (CAPABILITIES_ACTION, HEALTH_ACTION):
        _reject_unknown(document, _COMMON)
        return {"request_id": request_id}
    if action_name == STATUS_ACTION:
        _reject_unknown(document, _COMMON | {"jobId"})
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
    _validate_common(document, require_request_id=False)
    allowed = {
        "contract",
        "version",
        "jobId",
        "clientExecutionId",
        "fencingToken",
        "payloadId",
        "ordinal",
        "schemaId",
        "dataContractSha256",
        "rows",
    }
    _reject_unknown(document, allowed)
    schema_id = document.get("schemaId")
    if schema_id not in (FIT_SCHEMA_ID, PREDICT_SCHEMA_ID):
        raise invalid(
            f"schemaId must be one of: {FIT_SCHEMA_ID}, {PREDICT_SCHEMA_ID}"
        )
    return {
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "fencing_token": _fencing_token(document, "fencingToken"),
        "payload_id": _uuid(document, "payloadId"),
        "ordinal": _ordinal(document, "ordinal"),
        "schema_id": schema_id,
        "input_kind": "fit" if schema_id == FIT_SCHEMA_ID else "predict",
        "data_contract_sha256": _sha256(document, "dataContractSha256"),
        "rows": _nonnegative_integer(document, "rows"),
    }


@dataclass(frozen=True, slots=True)
class JobDataDescriptor:
    job_id: str
    ordinal: int


def parse_input_descriptor(descriptor: object) -> JobDataDescriptor:
    parts = _descriptor_parts(descriptor)
    if (
        len(parts) != 6
        or parts[:3] != ("transformer", CONTRACT_PATH_VERSION, "jobs")
        or parts[4] != "inputs"
    ):
        raise invalid("invalid input Flight descriptor")
    return JobDataDescriptor(
        job_id=_uuid_value(parts[3], "descriptor jobId"),
        ordinal=_ordinal_text(parts[5], "descriptor ordinal"),
    )


def parse_output_descriptor(descriptor: object) -> JobDataDescriptor:
    parts = _descriptor_parts(descriptor)
    if (
        len(parts) != 6
        or parts[:3] != ("transformer", CONTRACT_PATH_VERSION, "jobs")
        or parts[4] != "outputs"
    ):
        raise invalid("invalid output Flight descriptor")
    return JobDataDescriptor(
        job_id=_uuid_value(parts[3], "descriptor jobId"),
        ordinal=_ordinal_text(parts[5], "descriptor ordinal"),
    )


def response_document(request_id: str, **fields: object) -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "requestId": request_id,
        **{key: _json_value(value, key) for key, value in fields.items()},
    }


def encode_document(document: JsonObject | list[JsonValue]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_request_hash(document: JsonObject) -> str:
    canonical = {
        key: value
        for key, value in document.items()
        if key not in ("requestId", "idempotencyKey")
    }
    return hashlib.sha256(encode_document(canonical)).hexdigest()


def canonical_manifest_receipts(
    inputs: Iterable[object],
) -> list[dict[str, object]]:
    return canonical_receipts(inputs)


def canonical_manifest_hash(inputs: Iterable[object]) -> str:
    return manifest_sha256(inputs)


def model_config_to_api(
    config: ModelConfig | Mapping[str, object],
) -> JsonObject:
    if not isinstance(config, ModelConfig):
        parsed = ModelConfig.from_dict(config)
        if parsed is None:
            raise invalid("model configuration must not be empty")
        config = parsed
    return {
        "seqLen": config.seq_len,
        "hidden": config.hidden,
        "layers": config.layers,
        "dropout": config.dropout,
        "nhead": config.nhead,
        "mode": config.context_mode,
        "outDim": config.out_dim,
        "featureDim": config.feature_dim,
    }


def train_config_to_api(
    config: TrainConfig | Mapping[str, object],
) -> JsonObject:
    if not isinstance(config, TrainConfig):
        parsed = TrainConfig.from_dict(config)
        if parsed is None:
            raise invalid("training configuration must not be empty")
        config = parsed
    return train_config_to_manifest(config)


def data_contract_to_api(contract: Mapping[str, object]) -> JsonObject:
    return {
        "id": cast(JsonValue, contract.get("id")),
        "version": cast(JsonValue, contract.get("version")),
        "dataContractSha256": cast(JsonValue, contract.get(
            "data_contract_sha256",
            contract.get("dataContractSha256"),
        )),
        "seqLen": cast(JsonValue, contract.get("seq_len", contract.get("seqLen"))),
        "featureDim": cast(JsonValue, contract.get(
            "feature_dim",
            contract.get("featureDim"),
        )),
        "targetSchemaId": cast(JsonValue, contract.get(
            "target_schema_id",
            contract.get("targetSchemaId"),
        )),
    }


@overload
def _validate_common(
    document: Mapping[str, object],
    *,
    require_request_id: Literal[True] = True,
) -> str: ...


@overload
def _validate_common(
    document: Mapping[str, object],
    *,
    require_request_id: Literal[False],
) -> None: ...


def _validate_common(
    document: Mapping[str, object],
    *,
    require_request_id: bool = True,
) -> str | None:
    if document.get("contract") != CONTRACT_NAME:
        raise invalid(f"contract must be {CONTRACT_NAME!r}")
    version = document.get("version")
    if isinstance(version, bool) or version != CONTRACT_VERSION:
        raise invalid(f"version must be {CONTRACT_VERSION}")
    if not require_request_id:
        if "requestId" in document:
            raise invalid("DoPut metadata must not contain requestId")
        return None
    return _uuid(document, "requestId")


def _validate_create(
    document: JsonObject,
    request_id: str,
) -> CreateRequestFields:
    allowed = _MUTATING | {
        "jobId",
        "clientExecutionId",
        "operation",
        "device",
        "modelLabel",
        "modelRef",
        "modelAlias",
        "modelConfig",
        "trainingConfig",
        "predictionColumn",
        "dataContract",
        "mlContract",
    }
    _reject_unknown(document, allowed)
    operation = document.get("operation")
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise invalid(
            f"operation must be one of: {', '.join(SUPPORTED_OPERATIONS)}"
        )
    device = document.get("device")
    if not isinstance(device, str) or device not in SUPPORTED_DEVICES:
        raise invalid(f"device must be one of: {', '.join(SUPPORTED_DEVICES)}")
    prediction_column = document.get("predictionColumn", "out")
    if (
        not isinstance(prediction_column, str)
        or not _PREDICTION_COLUMN.fullmatch(prediction_column)
    ):
        raise invalid("predictionColumn has an invalid value")
    data_contract = _data_contract(document.get("dataContract"))
    requested_ml_contract = _ml_contract(document.get("mlContract"))

    common: CreateRequestFields = {
        "request_id": request_id,
        "idempotency_key": _idempotency_key(document),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "operation": operation,
        "device": device,
        "prediction_column": prediction_column,
        "data_contract": data_contract,
        "ml_contract": requested_ml_contract,
    }
    if operation == "fit":
        if any(key in document for key in ("modelRef", "modelAlias")):
            raise invalid("fit create does not accept modelRef or modelAlias")
        model_config = _model_config(document.get("modelConfig"))
        if model_config.seq_len != data_contract["seq_len"]:
            raise invalid("modelConfig.seqLen must match dataContract.seqLen")
        resolved_model_config = replace(
            model_config,
            feature_dim=data_contract["feature_dim"],
        )
        train_config = _train_config(
            document.get("trainingConfig", {})
        )
        if requested_ml_contract != ml_contract(train_config):
            raise invalid(
                "mlContract does not match the target-aligned training objective"
            )
        common["model_label"] = _label(document, "modelLabel")
        common["model_config"] = resolved_model_config
        common["train_config"] = train_config
    else:
        if any(
            key in document
            for key in ("modelLabel", "modelConfig", "trainingConfig")
        ):
            raise invalid(
                "predict create accepts a modelRef or modelAlias, not training config"
            )
        selectors = [key for key in ("modelRef", "modelAlias") if key in document]
        if len(selectors) != 1:
            raise invalid(
                "predict create requires exactly one of modelRef or modelAlias"
            )
        selector = selectors[0]
        common["model_ref"] = _label(document, selector)
        common["model_selector"] = selector
    return common


def _validate_acquire(
    document: JsonObject,
    request_id: str,
) -> AcquireRequestFields:
    _reject_unknown(
        document,
        _MUTATING
        | {
            "jobId",
            "previousClientExecutionId",
            "expectedFencingToken",
            "clientExecutionId",
        },
    )
    previous = _uuid(document, "previousClientExecutionId")
    current = _uuid(document, "clientExecutionId")
    if previous == current:
        raise invalid("clientExecutionId must change during acquire")
    return {
        "request_id": request_id,
        "idempotency_key": _idempotency_key(document),
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
    allowed = _COMMON | {
        "jobId",
        "afterRevision",
        "snapshotRevision",
        "cursor",
        "limit",
    }
    _reject_unknown(document, allowed)
    after_revision = _nonnegative_integer(document, "afterRevision")
    has_snapshot = "snapshotRevision" in document
    has_cursor = "cursor" in document
    if has_snapshot != has_cursor:
        raise invalid("snapshotRevision and cursor must be supplied together")
    snapshot = (
        _nonnegative_integer(document, "snapshotRevision")
        if has_snapshot
        else None
    )
    cursor = _nonnegative_integer(document, "cursor") if has_cursor else None
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
    _reject_unknown(
        document,
        _FENCED
        | {"payloadCount", "totalRows", "totalBytes", "manifestSha256"},
    )
    payload_count = _nonnegative_integer(document, "payloadCount")
    if payload_count > MAX_PAYLOADS_PER_JOB:
        raise invalid(
            f"payloadCount must not exceed {MAX_PAYLOADS_PER_JOB}"
        )
    return {
        "request_id": request_id,
        "idempotency_key": _idempotency_key(document),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "fencing_token": _fencing_token(document, "fencingToken"),
        "payload_count": payload_count,
        "total_rows": _nonnegative_integer(document, "totalRows"),
        "total_bytes": _nonnegative_integer(document, "totalBytes"),
        "manifest_sha256": _sha256(document, "manifestSha256"),
    }


def _validate_outputs_list(
    document: JsonObject,
    request_id: str,
) -> OutputsListRequestFields:
    _reject_unknown(document, _COMMON | {"jobId", "cursor", "limit"})
    return {
        "request_id": request_id,
        "job_id": _uuid(document, "jobId"),
        "cursor": (
            _nonnegative_integer(document, "cursor")
            if "cursor" in document
            else None
        ),
        "limit": _page_limit(document),
    }


def _validate_cancel(
    document: JsonObject,
    request_id: str,
) -> CancelRequestFields:
    _reject_unknown(document, _FENCED)
    return {
        "request_id": request_id,
        "idempotency_key": _idempotency_key(document),
        "job_id": _uuid(document, "jobId"),
        "client_execution_id": _uuid(document, "clientExecutionId"),
        "fencing_token": _fencing_token(document, "fencingToken"),
    }


def _validate_model_describe(
    document: JsonObject,
    request_id: str,
) -> ModelDescribeRequestFields:
    _reject_unknown(document, _COMMON | {"modelRef", "modelAlias"})
    selectors = [key for key in ("modelRef", "modelAlias") if key in document]
    if len(selectors) != 1:
        raise invalid(
            "model.describe requires exactly one of modelRef or modelAlias"
        )
    selector = selectors[0]
    return {
        "request_id": request_id,
        "model_ref": _label(document, selector),
        "model_selector": selector,
    }


def _data_contract(document: object) -> DataContractFields:
    if not isinstance(document, dict):
        raise invalid("dataContract must be an object")
    values = cast(dict[str, object], document)
    allowed = {
        "id",
        "version",
        "dataContractSha256",
        "seqLen",
        "featureDim",
        "targetSchemaId",
    }
    _reject_unknown(values, allowed, "dataContract")
    result: DataContractFields = {
        "id": _label(values, "id"),
        "version": _positive_integer(values, "version"),
        "data_contract_sha256": _sha256(values, "dataContractSha256"),
        "seq_len": _positive_integer(values, "seqLen"),
        "feature_dim": _positive_integer(values, "featureDim"),
        "target_schema_id": _label(values, "targetSchemaId"),
    }
    if result["target_schema_id"] != TARGET_SCHEMA_ID:
        raise invalid(f"dataContract.targetSchemaId must be {TARGET_SCHEMA_ID}")
    return result


def _ml_contract(document: object) -> MlContractFields:
    if not isinstance(document, dict):
        raise invalid("mlContract must be an object")
    values = cast(dict[str, object], document)
    allowed = {
        "targetSchemaId",
        "predictionSchemaId",
        "objectiveId",
        "objectiveConfigSha256",
        "checkpointFormat",
        "targetWidth",
        "predictionSpace",
    }
    _reject_unknown(values, allowed, "mlContract")
    expected: dict[str, str | int] = {
        "targetSchemaId": TARGET_SCHEMA_ID,
        "predictionSchemaId": PREDICTION_SCHEMA_ID,
        "objectiveId": OBJECTIVE_ID,
        "checkpointFormat": CHECKPOINT_FORMAT,
        "targetWidth": TARGET_WIDTH,
        "predictionSpace": "target",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise invalid(f"mlContract.{key} must be {value!r}")
    return {
        "targetSchemaId": TARGET_SCHEMA_ID,
        "predictionSchemaId": PREDICTION_SCHEMA_ID,
        "objectiveId": OBJECTIVE_ID,
        "objectiveConfigSha256": _sha256(values, "objectiveConfigSha256"),
        "checkpointFormat": CHECKPOINT_FORMAT,
        "targetWidth": TARGET_WIDTH,
        "predictionSpace": "target",
    }


def _model_config(document: object) -> ModelConfig:
    if not isinstance(document, dict):
        raise invalid("modelConfig must be an object")
    values = cast(dict[str, object], document)
    allowed = {"seqLen", "hidden", "layers", "dropout", "nhead", "mode"}
    _reject_unknown(values, allowed, "modelConfig")
    if "seqLen" not in values:
        raise invalid("modelConfig.seqLen is required")
    mapped: dict[str, object] = {
        "seq_len": values.get("seqLen"),
        "hidden": values.get(
            "hidden",
            ModelConfig.__dataclass_fields__["hidden"].default,
        ),
        "layers": values.get(
            "layers",
            ModelConfig.__dataclass_fields__["layers"].default,
        ),
        "dropout": values.get(
            "dropout",
            ModelConfig.__dataclass_fields__["dropout"].default,
        ),
        "nhead": values.get(
            "nhead",
            ModelConfig.__dataclass_fields__["nhead"].default,
        ),
        "context_mode": values.get(
            "mode",
            ModelConfig.__dataclass_fields__["context_mode"].default,
        ),
    }
    _require_numbers(mapped, integer=("seq_len", "hidden", "layers", "nhead"))
    try:
        config = ModelConfig.from_dict(mapped)
        if config is None:
            raise ValueError("model configuration must not be empty")
        return config
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid modelConfig: {exc}") from exc


def _train_config(document: object) -> TrainConfig:
    if not isinstance(document, dict):
        raise invalid("trainingConfig must be an object")
    values = cast(dict[str, object], document)
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
    _reject_unknown(values, set(names), "trainingConfig")
    mapped: dict[str, object] = {
        names[key]: value for key, value in values.items()
    }
    for name in ("use_amp", "deterministic"):
        if name in mapped and not isinstance(mapped[name], bool):
            raise invalid(
                f"trainingConfig.{_api_name(names, name)} must be a boolean"
            )
    _require_numbers(
        mapped,
        integer=(
            "batch_size",
            "epochs",
            "loss_stage",
            "stage_size",
            "seed",
        ),
    )
    try:
        if not mapped:
            return TrainConfig()
        config = TrainConfig.from_dict(mapped)
        if config is None:
            raise ValueError("training configuration must not be empty")
        return config
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid trainingConfig: {exc}") from exc


def _require_numbers(
    mapping: Mapping[str, object],
    *,
    integer: tuple[str, ...],
) -> None:
    for key, value in mapping.items():
        if key in integer:
            if isinstance(value, bool) or not isinstance(value, int):
                raise invalid(f"{key} must be an integer")
        elif key in (
            "dropout",
            "lr",
            "weight_decay",
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise invalid(f"{key} must be a number")
            if not math.isfinite(value):
                raise invalid(f"{key} must be finite")


def _descriptor_parts(descriptor: object) -> tuple[str, ...]:
    path = cast(object, getattr(descriptor, "path", None))
    if path is None:
        raise invalid("Flight descriptor must be a path descriptor")
    if not isinstance(path, Sequence):
        raise invalid("Flight descriptor path must be a sequence")
    parts: list[str] = []
    for value in cast(Sequence[object], path):
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise invalid("Flight descriptor path must be UTF-8") from exc
        if not isinstance(value, str):
            raise invalid("Flight descriptor path must contain strings")
        parts.append(value)
    return tuple(parts)


def _reject_unknown(
    document: Mapping[str, object],
    allowed: Set[str],
    location: str = "request",
) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise invalid(f"unknown {location} field(s): {', '.join(unknown)}")


def _uuid(document: Mapping[str, object], key: str) -> str:
    if key not in document:
        raise invalid(f"{key} is required")
    return _uuid_value(document[key], key)


def _uuid_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise invalid(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise invalid(f"{label} must be a UUID string") from exc
    if str(parsed) != value.lower():
        raise invalid(f"{label} must be a canonical UUID string")
    return str(parsed)


def _idempotency_key(document: Mapping[str, object]) -> str:
    value = document.get("idempotencyKey")
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY.fullmatch(value):
        raise invalid("idempotencyKey has an invalid value")
    return value


def _label(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise invalid(f"{key} has an invalid value")
    return value


def _sha256(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise invalid(f"{key} must be a lowercase SHA-256 digest")
    return value


def _fencing_token(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, str) or not _FENCING_TOKEN.fullmatch(value):
        raise invalid(f"{key} must be a positive canonical decimal string")
    parsed = int(value)
    if parsed > 2**63 - 1:
        raise invalid(f"{key} exceeds the supported range")
    return parsed


def _nonnegative_integer(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise invalid(f"{key} must be a non-negative integer")
    return value


def _positive_integer(document: Mapping[str, object], key: str) -> int:
    value = _nonnegative_integer(document, key)
    if value == 0:
        raise invalid(f"{key} must be a positive integer")
    return value


def _ordinal(document: Mapping[str, object], key: str) -> int:
    value = _nonnegative_integer(document, key)
    if value >= MAX_PAYLOADS_PER_JOB:
        raise invalid(f"{key} must be less than {MAX_PAYLOADS_PER_JOB}")
    return value


def _nonnegative_text_integer(value: str, label: str) -> int:
    if (
        not value
        or len(value) > 20
        or not value.isascii()
        or not value.isdigit()
    ):
        raise invalid(f"{label} must be a non-negative integer")
    parsed = int(value)
    if str(parsed) != value:
        raise invalid(f"{label} must use canonical decimal notation")
    return parsed


def _ordinal_text(value: str, label: str) -> int:
    parsed = _nonnegative_text_integer(value, label)
    if parsed >= MAX_PAYLOADS_PER_JOB:
        raise invalid(f"{label} must be less than {MAX_PAYLOADS_PER_JOB}")
    return parsed


def _page_limit(document: Mapping[str, object]) -> int:
    if "limit" not in document:
        return MAX_PAGE_ITEMS
    limit = _positive_integer(document, "limit")
    if limit > MAX_PAGE_ITEMS:
        raise invalid(f"limit must not exceed {MAX_PAGE_ITEMS}")
    return limit


def _api_name(mapping: dict[str, str], internal: str) -> str:
    return next(key for key, value in mapping.items() if value == internal)


def _json_value(value: object, location: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError(f"{location} contains a non-string object key")
        return {
            cast(str, key): _json_value(item, f"{location}.{key}")
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(cast(Sequence[object], value))
        ]
    raise TypeError(f"{location} is not JSON-serializable")


__all__ = [
    "JobDataDescriptor",
    "canonical_manifest_hash",
    "canonical_manifest_receipts",
    "canonical_request_hash",
    "data_contract_to_api",
    "encode_document",
    "model_config_to_api",
    "parse_action_body",
    "parse_input_descriptor",
    "parse_output_descriptor",
    "response_document",
    "train_config_to_api",
    "validate_action_request",
    "validate_upload_metadata",
]
