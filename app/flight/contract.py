import hashlib
import json
import math
import re
import uuid

from app.flight.constants import (
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CONTRACT_VERSION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    MAX_MANIFEST_ITEMS,
    PREDICT_SCHEMA_ID,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
    SUPPORTED_DEVICES,
    SUPPORTED_OPERATIONS,
)
from app.flight.errors import invalid
from app.training.run_config import ModelConfig, TrainConfig

MAX_ACTION_DOCUMENT_BYTES = 64 * 1024
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PREDICTION_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_COMMON = {"contract", "version", "requestId"}
_MUTATING = _COMMON | {"idempotencyKey"}


def parse_action_body(body) -> dict:
    if hasattr(body, "to_pybytes"):
        body = body.to_pybytes()
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise invalid("action body must be UTF-8 JSON bytes")
    body = bytes(body)
    if len(body) > MAX_ACTION_DOCUMENT_BYTES:
        raise invalid(
            f"action body exceeds {MAX_ACTION_DOCUMENT_BYTES} bytes"
        )
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise invalid("action body must be a valid UTF-8 JSON document") from exc
    if not isinstance(document, dict):
        raise invalid("action body must be a JSON object")
    return document


def validate_action_request(action_name: str, document: dict) -> dict:
    request_id = _validate_common(document)
    if action_name in (CAPABILITIES_ACTION, HEALTH_ACTION):
        _reject_unknown(document, _COMMON)
        return {"request_id": request_id}
    if action_name == STATUS_ACTION:
        _reject_unknown(document, _COMMON | {"jobId"})
        return {"request_id": request_id, "job_id": _uuid(document, "jobId")}
    if action_name == CREATE_ACTION:
        return _validate_create(document, request_id)
    if action_name == SEAL_ACTION:
        return _validate_seal(document, request_id)
    if action_name in (START_ACTION, CANCEL_ACTION):
        _reject_unknown(document, _MUTATING | {"jobId"})
        return {
            "request_id": request_id,
            "idempotency_key": _idempotency_key(document),
            "job_id": _uuid(document, "jobId"),
        }
    raise invalid(f"unsupported action: {action_name}")


def validate_upload_metadata(document: dict) -> dict:
    request_id = _validate_common(document, require_request_id=False)
    allowed = _COMMON | {"jobId", "payloadId", "ordinal", "schemaId", "rows"}
    _reject_unknown(document, allowed)
    ordinal = _ordinal(document, "ordinal")
    rows = _nonnegative_integer(document, "rows")
    schema_id = document.get("schemaId")
    if schema_id not in (FIT_SCHEMA_ID, PREDICT_SCHEMA_ID):
        raise invalid(
            f"schemaId must be one of: {FIT_SCHEMA_ID}, {PREDICT_SCHEMA_ID}"
        )
    return {
        "request_id": request_id,
        "job_id": _uuid(document, "jobId"),
        "payload_id": _uuid(document, "payloadId"),
        "ordinal": ordinal,
        "schema_id": schema_id,
        "rows": rows,
    }


def parse_input_descriptor(descriptor) -> tuple[str, int]:
    parts = _descriptor_parts(descriptor)
    if len(parts) != 6 or parts[:3] != ("transformer", "v1", "jobs"):
        raise invalid("invalid input Flight descriptor")
    if parts[4] != "inputs":
        raise invalid("invalid input Flight descriptor")
    job_id = _uuid_value(parts[3], "descriptor jobId")
    ordinal = _ordinal_text(parts[5], "descriptor ordinal")
    return job_id, ordinal


def parse_output_descriptor(descriptor) -> tuple[str, int]:
    parts = _descriptor_parts(descriptor)
    if len(parts) != 6 or parts[:3] != ("transformer", "v1", "jobs"):
        raise invalid("invalid output Flight descriptor")
    if parts[4] != "outputs":
        raise invalid("invalid output Flight descriptor")
    return (
        _uuid_value(parts[3], "descriptor jobId"),
        _ordinal_text(parts[5], "descriptor ordinal"),
    )


def response_document(request_id: str, **fields) -> dict:
    return {
        "contract": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "requestId": request_id,
        **fields,
    }


def encode_document(document: dict) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_request_hash(document: dict) -> str:
    canonical = {
        key: value
        for key, value in document.items()
        if key not in ("requestId", "idempotencyKey")
    }
    return hashlib.sha256(encode_document(canonical)).hexdigest()


def canonical_manifest_hash(manifest: list[dict]) -> str:
    return hashlib.sha256(encode_document(manifest)).hexdigest()


def model_config_to_api(config: ModelConfig | dict) -> dict:
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
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


def train_config_to_api(config: TrainConfig | dict) -> dict:
    if isinstance(config, dict):
        config = TrainConfig.from_dict(config)
    return {
        "lr": config.lr,
        "batchSize": config.batch_size,
        "epochs": config.epochs,
        "patience": config.patience,
        "lossStage": config.loss_stage,
        "lossSchedule": config.loss_schedule,
        "stageSize": config.stage_size,
        "useAmp": config.use_amp,
        "weightDecay": config.weight_decay,
        "monitor": config.monitor,
        "monitorMinImprovement": config.monitor_min_improvement,
        "saveBestCheckpoint": config.save_best_checkpoint,
        "seed": config.seed,
        "deterministic": config.deterministic,
    }


def _validate_common(document: dict, *, require_request_id: bool = True) -> str | None:
    if document.get("contract") != CONTRACT_NAME:
        raise invalid(f"contract must be {CONTRACT_NAME!r}")
    version = document.get("version")
    if isinstance(version, bool) or version != CONTRACT_VERSION:
        raise invalid(f"version must be {CONTRACT_VERSION}")
    if not require_request_id and "requestId" not in document:
        return None
    return _uuid(document, "requestId")


def _validate_create(document: dict, request_id: str) -> dict:
    allowed = _MUTATING | {
        "operation",
        "device",
        "modelLabel",
        "modelRef",
        "modelAlias",
        "modelConfig",
        "trainingConfig",
        "predictionColumn",
    }
    _reject_unknown(document, allowed)
    operation = document.get("operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise invalid(f"operation must be one of: {', '.join(SUPPORTED_OPERATIONS)}")
    device = document.get("device")
    if device not in SUPPORTED_DEVICES:
        raise invalid(f"device must be one of: {', '.join(SUPPORTED_DEVICES)}")
    prediction_column = document.get("predictionColumn", "out")
    if not isinstance(prediction_column, str) or not _PREDICTION_COLUMN.fullmatch(
        prediction_column
    ):
        raise invalid("predictionColumn has an invalid value")

    result = {
        "request_id": request_id,
        "idempotency_key": _idempotency_key(document),
        "operation": operation,
        "device": device,
        "prediction_column": prediction_column,
    }
    if operation == "fit":
        prohibited = ("modelRef", "modelAlias")
        if any(key in document for key in prohibited):
            raise invalid("fit create does not accept modelRef or modelAlias")
        result["model_label"] = _label(document, "modelLabel")
        result["model_config"] = _model_config(document.get("modelConfig"))
        result["train_config"] = _train_config(document.get("trainingConfig", {}))
    else:
        prohibited = ("modelLabel", "modelConfig", "trainingConfig")
        if any(key in document for key in prohibited):
            raise invalid(
                "predict create accepts a modelRef or modelAlias, not training config"
            )
        selectors = [key for key in ("modelRef", "modelAlias") if key in document]
        if len(selectors) != 1:
            raise invalid("predict create requires exactly one of modelRef or modelAlias")
        selector = selectors[0]
        result["model_ref"] = _label(document, selector)
        result["model_selector"] = selector
    return result


def _validate_seal(document: dict, request_id: str) -> dict:
    _reject_unknown(document, _MUTATING | {"jobId", "manifest"})
    manifest = document.get("manifest")
    if not isinstance(manifest, list):
        raise invalid("manifest must be an array")
    if len(manifest) > MAX_MANIFEST_ITEMS:
        raise invalid(
            f"manifest must contain at most {MAX_MANIFEST_ITEMS} entries"
        )
    parsed = []
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            raise invalid(f"manifest item {index} must be an object")
        _reject_unknown(item, {"payloadId", "ordinal", "sha256"}, f"manifest item {index}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise invalid(f"manifest item {index} has invalid sha256")
        parsed.append({
            "payloadId": _uuid(item, "payloadId"),
            "ordinal": _ordinal(item, "ordinal"),
            "sha256": digest,
        })
    return {
        "request_id": request_id,
        "idempotency_key": _idempotency_key(document),
        "job_id": _uuid(document, "jobId"),
        "manifest": parsed,
    }


def _model_config(document) -> ModelConfig:
    if not isinstance(document, dict):
        raise invalid("modelConfig must be an object")
    allowed = {"seqLen", "hidden", "layers", "dropout", "nhead", "mode"}
    _reject_unknown(document, allowed, "modelConfig")
    if "seqLen" not in document:
        raise invalid("modelConfig.seqLen is required")
    mapped = {
        "seq_len": document.get("seqLen"),
        "hidden": document.get("hidden", ModelConfig.__dataclass_fields__["hidden"].default),
        "layers": document.get("layers", ModelConfig.__dataclass_fields__["layers"].default),
        "dropout": document.get("dropout", ModelConfig.__dataclass_fields__["dropout"].default),
        "nhead": document.get("nhead", ModelConfig.__dataclass_fields__["nhead"].default),
        "context_mode": document.get("mode", ModelConfig.__dataclass_fields__["context_mode"].default),
    }
    _require_numbers(mapped, integer=("seq_len", "hidden", "layers", "nhead"))
    try:
        return ModelConfig(**mapped)
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid modelConfig: {exc}") from exc


def _train_config(document) -> TrainConfig:
    if not isinstance(document, dict):
        raise invalid("trainingConfig must be an object")
    names = {
        "lr": "lr",
        "batchSize": "batch_size",
        "epochs": "epochs",
        "patience": "patience",
        "lossStage": "loss_stage",
        "lossSchedule": "loss_schedule",
        "stageSize": "stage_size",
        "useAmp": "use_amp",
        "weightDecay": "weight_decay",
        "monitor": "monitor",
        "monitorMinImprovement": "monitor_min_improvement",
        "saveBestCheckpoint": "save_best_checkpoint",
        "seed": "seed",
        "deterministic": "deterministic",
    }
    _reject_unknown(document, set(names), "trainingConfig")
    mapped = {names[key]: value for key, value in document.items()}
    for name in ("use_amp", "save_best_checkpoint", "deterministic"):
        if name in mapped and not isinstance(mapped[name], bool):
            raise invalid(f"trainingConfig.{_api_name(names, name)} must be a boolean")
    _require_numbers(
        mapped,
        integer=("batch_size", "epochs", "patience", "loss_stage", "stage_size", "seed"),
    )
    try:
        return TrainConfig(**mapped)
    except (TypeError, ValueError) as exc:
        raise invalid(f"invalid trainingConfig: {exc}") from exc


def _require_numbers(mapping: dict, *, integer: tuple[str, ...]) -> None:
    for key, value in mapping.items():
        if key in integer:
            if isinstance(value, bool) or not isinstance(value, int):
                raise invalid(f"{key} must be an integer")
        elif key in ("dropout", "lr", "weight_decay", "monitor_min_improvement"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise invalid(f"{key} must be a number")
            if not math.isfinite(value):
                raise invalid(f"{key} must be finite")


def _descriptor_parts(descriptor) -> tuple[str, ...]:
    path = getattr(descriptor, "path", None)
    if path is None:
        raise invalid("Flight descriptor must be a path descriptor")
    parts = []
    for value in path:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise invalid("Flight descriptor path must be UTF-8") from exc
        if not isinstance(value, str):
            raise invalid("Flight descriptor path must contain strings")
        parts.append(value)
    return tuple(parts)


def _reject_unknown(document: dict, allowed: set, location: str = "request") -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise invalid(f"unknown {location} field(s): {', '.join(unknown)}")


def _uuid(document: dict, key: str) -> str:
    if key not in document:
        raise invalid(f"{key} is required")
    return _uuid_value(document[key], key)


def _uuid_value(value, label: str) -> str:
    if not isinstance(value, str):
        raise invalid(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise invalid(f"{label} must be a UUID string") from exc
    if str(parsed) != value.lower():
        raise invalid(f"{label} must be a canonical UUID string")
    return str(parsed)


def _idempotency_key(document: dict) -> str:
    value = document.get("idempotencyKey")
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY.fullmatch(value):
        raise invalid("idempotencyKey has an invalid value")
    return value


def _label(document: dict, key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise invalid(f"{key} has an invalid value")
    return value


def _nonnegative_integer(document: dict, key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise invalid(f"{key} must be a non-negative integer")
    return value


def _ordinal(document: dict, key: str) -> int:
    value = _nonnegative_integer(document, key)
    if value >= MAX_MANIFEST_ITEMS:
        raise invalid(f"{key} must be less than {MAX_MANIFEST_ITEMS}")
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
    if parsed >= MAX_MANIFEST_ITEMS:
        raise invalid(f"{label} must be less than {MAX_MANIFEST_ITEMS}")
    return parsed


def _api_name(mapping: dict[str, str], internal: str) -> str:
    return next(key for key, value in mapping.items() if value == internal)
