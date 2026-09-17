from __future__ import annotations

from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.training_telemetry.v3 import (
    MAX_RESPONSE_BYTES,
    SNAPSHOT_CAPACITY_RETRY_AFTER_SECONDS,
    validate_training_telemetry_document,
)
from app.service.adapters.inbound.flight.documents import encode_document
from app.service.application.messages.training_telemetry import (
    GradientInteractionsResult,
    TrainingTelemetryReportResult,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode


def present_training_telemetry_report(
    result: TrainingTelemetryReportResult,
) -> JsonObject:
    common: JsonObject = {
        "requestId": result.request_id,
        "state": result.state,
        "modelRef": result.model_ref,
    }
    if result.state == "pending":
        return {
            **common,
            "reason": "MATERIALIZATION_PENDING",
            "retryAfterSeconds": 5,
        }
    if result.state == "unavailable":
        return {
            **common,
            "reason": result.unavailable_reason,
        }
    if any(
        value is None
        for value in (
            result.producing_run_id,
            result.semantic_digests,
            result.layout,
            result.coverage,
            result.selection,
            result.health_totals,
            result.gradient_interactions,
        )
    ):
        raise RuntimeError("available training telemetry result is incomplete")
    return {
        **common,
        "producingRunId": result.producing_run_id,
        "dataContractSha256": cast(
            str,
            cast(JsonObject, result.semantic_digests)["dataContractSha256"],
        ),
        "modelDefinitionSha256": cast(
            str,
            cast(JsonObject, result.semantic_digests)["modelDefinitionSha256"],
        ),
        "layout": dict(cast(JsonObject, result.layout)),
        "coverage": dict(cast(JsonObject, result.coverage)),
        "selection": dict(cast(JsonObject, result.selection)),
        "anchors": [dict(item) for item in result.anchors],
        "healthTotals": dict(cast(JsonObject, result.health_totals)),
        "gradientInteractions": dict(
            cast(JsonObject, result.gradient_interactions)
        ),
        "epochPage": {
            "items": [dict(item) for item in result.epoch_items],
            "nextCursor": result.next_cursor,
            "cursorExpiresAt": result.cursor_expires_at,
        },
    }


def present_gradient_interactions(
    result: GradientInteractionsResult,
) -> JsonObject:
    common: JsonObject = {
        "requestId": result.request_id,
        "state": result.state,
        "modelRef": result.model_ref,
        "epoch": result.epoch,
    }
    if result.state == "notCollected":
        return {**common, "reason": result.reason}
    if result.producing_run_id is None:
        raise RuntimeError("available gradient interaction result is incomplete")
    return {
        **common,
        "producingRunId": result.producing_run_id,
        "components": [dict(item) for item in result.components],
        "pairPage": {
            "items": [dict(item) for item in result.pair_items],
            "nextCursor": result.next_cursor,
            "cursorExpiresAt": result.cursor_expires_at,
        },
    }


def training_telemetry_response(document: JsonObject, schema_name: str) -> bytes:
    validate_training_telemetry_document(document, schema_name)
    encoded = encode_document(document)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise telemetry_error(
            ErrorCode.RESOURCE_EXHAUSTED,
            "TELEMETRY_RESPONSE_BUDGET_EXCEEDED",
            "training telemetry response exceeds the configured budget",
            operation=(
                "report"
                if schema_name == "report-result"
                else "gradientInteractions"
            ),
            maxResponseBytes=MAX_RESPONSE_BYTES,
            actualResponseBytes=len(encoded),
        )
    return encoded


def telemetry_error(
    code: ErrorCode,
    reason: str,
    message: str,
    **fields: object,
) -> ServiceError:
    detail = cast(JsonObject, {
        "code": code.value,
        "reason": reason,
        **fields,
        "message": message,
    })
    validate_training_telemetry_document(detail, "error-detail")
    return ServiceError(code, message, detail=detail)


def invalid_telemetry_query(message: str, path: str) -> ServiceError:
    return telemetry_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_TELEMETRY_QUERY",
        message,
        path=path,
    )


def invalid_telemetry_cursor() -> ServiceError:
    return telemetry_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_TELEMETRY_CURSOR",
        "training telemetry cursor is invalid",
        path="/cursor",
    )


def expired_telemetry_cursor() -> ServiceError:
    return telemetry_error(
        ErrorCode.FAILED_PRECONDITION,
        "TELEMETRY_CURSOR_EXPIRED",
        "training telemetry cursor has expired",
        path="/cursor",
        restartRequired=True,
    )


def invalidated_telemetry_cursor() -> ServiceError:
    return telemetry_error(
        ErrorCode.FAILED_PRECONDITION,
        "TELEMETRY_CURSOR_INVALIDATED",
        "training telemetry cursor belongs to a previous service instance",
        path="/cursor",
        restartRequired=True,
    )


def telemetry_model_not_found(model_ref: str) -> ServiceError:
    return telemetry_error(
        ErrorCode.NOT_FOUND,
        "MODEL_NOT_FOUND",
        "model generation was not found",
        modelRef=model_ref,
    )


def telemetry_stored_metadata_invalid(model_ref: str, path: str) -> ServiceError:
    return telemetry_error(
        ErrorCode.MODEL_CORRUPT,
        "STORED_MODEL_METADATA_INVALID",
        "stored model metadata is invalid",
        modelRef=model_ref,
        path=path,
    )


def telemetry_integrity_failed(model_ref: str, path: str) -> ServiceError:
    return telemetry_error(
        ErrorCode.TELEMETRY_CORRUPT,
        "TRAINING_TELEMETRY_INTEGRITY_FAILED",
        "training telemetry integrity check failed",
        modelRef=model_ref,
        path=path,
    )


def telemetry_backend_unavailable(model_ref: str) -> ServiceError:
    return telemetry_error(
        ErrorCode.UNAVAILABLE,
        "TRAINING_TELEMETRY_BACKEND_UNAVAILABLE",
        "training telemetry backend is temporarily unavailable",
        modelRef=model_ref,
    )


def telemetry_snapshot_capacity_exhausted(operation: str) -> ServiceError:
    return telemetry_error(
        ErrorCode.RESOURCE_EXHAUSTED,
        "TELEMETRY_SNAPSHOT_CAPACITY_EXHAUSTED",
        "training telemetry snapshot capacity is exhausted",
        operation=operation,
        retryable=True,
        retryAfterSeconds=SNAPSHOT_CAPACITY_RETRY_AFTER_SECONDS,
    )


__all__ = [
    "expired_telemetry_cursor",
    "invalid_telemetry_cursor",
    "invalid_telemetry_query",
    "invalidated_telemetry_cursor",
    "present_gradient_interactions",
    "present_training_telemetry_report",
    "telemetry_backend_unavailable",
    "telemetry_integrity_failed",
    "telemetry_model_not_found",
    "telemetry_snapshot_capacity_exhausted",
    "telemetry_stored_metadata_invalid",
    "training_telemetry_response",
]
