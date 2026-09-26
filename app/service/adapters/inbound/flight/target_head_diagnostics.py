from __future__ import annotations

from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.target_head_diagnostics.v5 import (
    MAX_RESPONSE_BYTES,
    SNAPSHOT_CAPACITY_RETRY_AFTER_SECONDS,
    validate_target_head_diagnostics_document,
)
from app.service.adapters.inbound.flight.documents import encode_document
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode


def target_head_diagnostics_response(document: JsonObject) -> bytes:
    validate_target_head_diagnostics_document(document, "report-result")
    encoded = encode_document(document)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise target_head_diagnostics_error(
            ErrorCode.RESOURCE_EXHAUSTED,
            "TARGET_HEAD_DIAGNOSTICS_RESPONSE_BUDGET_EXCEEDED",
            "ответ диагностики выходной головки превышает допустимый размер",
            maxResponseBytes=MAX_RESPONSE_BYTES,
            actualResponseBytes=len(encoded),
        )
    return encoded


def target_head_diagnostics_error(
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
    validate_target_head_diagnostics_document(detail, "error-detail")
    return ServiceError(code, message, detail=detail)


def invalid_target_head_diagnostics_query(
    message: str,
    path: str,
) -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_TARGET_HEAD_DIAGNOSTICS_QUERY",
        message,
        path=path,
    )


def invalid_target_head_diagnostics_cursor() -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_TARGET_HEAD_DIAGNOSTICS_CURSOR",
        "курсор диагностики выходной головки недействителен",
        path="/cursor",
    )


def expired_target_head_diagnostics_cursor() -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.FAILED_PRECONDITION,
        "TARGET_HEAD_DIAGNOSTICS_CURSOR_EXPIRED",
        "срок действия курсора диагностики выходной головки истёк",
        path="/cursor",
        restartRequired=True,
    )


def invalidated_target_head_diagnostics_cursor() -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.FAILED_PRECONDITION,
        "TARGET_HEAD_DIAGNOSTICS_CURSOR_INVALIDATED",
        "курсор диагностики выходной головки принадлежит прежнему запуску сервиса",
        path="/cursor",
        restartRequired=True,
    )


def target_head_diagnostics_query_unavailable() -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.FAILED_PRECONDITION,
        "TARGET_HEAD_DIAGNOSTICS_QUERY_UNAVAILABLE",
        "запрос диагностики выходной головки недоступен",
    )


def target_head_diagnostics_model_not_found(model_ref: str) -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.NOT_FOUND,
        "MODEL_NOT_FOUND",
        "generation модели не найдена",
        modelRef=model_ref,
    )


def target_head_diagnostics_stored_metadata_invalid(
    model_ref: str,
    path: str,
) -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.MODEL_CORRUPT,
        "STORED_MODEL_METADATA_INVALID",
        "сохранённая metadata модели недействительна",
        modelRef=model_ref,
        path=path,
    )


def target_head_diagnostics_integrity_failed(
    model_ref: str,
    path: str,
) -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.TARGET_HEAD_DIAGNOSTICS_CORRUPT,
        "TARGET_HEAD_DIAGNOSTICS_INTEGRITY_FAILED",
        "сохранённая диагностика выходной головки противоречива",
        modelRef=model_ref,
        path=path,
    )


def target_head_diagnostics_backend_unavailable(model_ref: str) -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.UNAVAILABLE,
        "TARGET_HEAD_DIAGNOSTICS_BACKEND_UNAVAILABLE",
        "хранилище диагностики выходной головки временно недоступно",
        modelRef=model_ref,
    )


def target_head_diagnostics_snapshot_capacity_exhausted() -> ServiceError:
    return target_head_diagnostics_error(
        ErrorCode.RESOURCE_EXHAUSTED,
        "TARGET_HEAD_DIAGNOSTICS_SNAPSHOT_CAPACITY_EXHAUSTED",
        "недостаточно capacity для snapshot диагностики выходной головки",
        retryable=True,
        retryAfterSeconds=SNAPSHOT_CAPACITY_RETRY_AFTER_SECONDS,
    )


__all__ = [
    "expired_target_head_diagnostics_cursor",
    "invalid_target_head_diagnostics_cursor",
    "invalid_target_head_diagnostics_query",
    "invalidated_target_head_diagnostics_cursor",
    "target_head_diagnostics_backend_unavailable",
    "target_head_diagnostics_error",
    "target_head_diagnostics_integrity_failed",
    "target_head_diagnostics_model_not_found",
    "target_head_diagnostics_query_unavailable",
    "target_head_diagnostics_response",
    "target_head_diagnostics_snapshot_capacity_exhausted",
    "target_head_diagnostics_stored_metadata_invalid",
]
