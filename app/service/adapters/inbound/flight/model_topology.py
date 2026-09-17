from __future__ import annotations

from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.model_topology.v1 import (
    MAX_RESPONSE_BYTES,
    validate_model_topology_document,
)
from app.service.adapters.inbound.flight.documents import encode_document
from app.service.application.messages.model_topology import ModelTopologyResult
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode


def present_model_topology(result: ModelTopologyResult) -> JsonObject:
    return {
        "requestId": result.request_id,
        "modelRef": result.model_ref,
        "modelDefinitionSha256": result.model_definition_sha256,
        "topologyRevision": 1,
        "nodes": [dict(node) for node in result.nodes],
        "edges": [dict(edge) for edge in result.edges],
    }


def model_topology_response(document: JsonObject) -> bytes:
    validate_model_topology_document(document, "detail-result")
    encoded = encode_document(document)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise topology_response_budget_exceeded(cast(str, document["modelRef"]))
    return encoded


def topology_error(
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
    validate_model_topology_document(detail, "error-detail")
    return ServiceError(code, message, detail=detail)


def invalid_model_topology_query(message: str, path: str) -> ServiceError:
    return topology_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_MODEL_TOPOLOGY_QUERY",
        message,
        path=path,
    )


def model_topology_unavailable() -> ServiceError:
    return topology_error(
        ErrorCode.FAILED_PRECONDITION,
        "MODEL_TOPOLOGY_QUERY_UNAVAILABLE",
        "model topology query is unavailable",
    )


def topology_model_not_found(model_ref: str) -> ServiceError:
    return topology_error(
        ErrorCode.NOT_FOUND,
        "MODEL_NOT_FOUND",
        "model generation was not found",
        modelRef=model_ref,
    )


def topology_stored_metadata_invalid(model_ref: str, path: str) -> ServiceError:
    return topology_error(
        ErrorCode.MODEL_CORRUPT,
        "STORED_MODEL_METADATA_INVALID",
        "stored model metadata is invalid",
        modelRef=model_ref,
        path=path,
    )


def topology_invalid(model_ref: str, path: str) -> ServiceError:
    return topology_error(
        ErrorCode.MODEL_CORRUPT,
        "MODEL_TOPOLOGY_INVALID",
        "model topology is invalid",
        modelRef=model_ref,
        path=path,
    )


def topology_response_budget_exceeded(model_ref: str) -> ServiceError:
    return topology_error(
        ErrorCode.RESOURCE_EXHAUSTED,
        "MODEL_TOPOLOGY_RESPONSE_BUDGET_EXCEEDED",
        "model topology response exceeds the configured budget",
        modelRef=model_ref,
        maxResponseBytes=MAX_RESPONSE_BYTES,
    )


def topology_registry_unavailable() -> ServiceError:
    return topology_error(
        ErrorCode.UNAVAILABLE,
        "MODEL_REGISTRY_UNAVAILABLE",
        "model registry is temporarily unavailable",
    )


__all__ = [
    "invalid_model_topology_query",
    "model_topology_response",
    "model_topology_unavailable",
    "present_model_topology",
    "topology_invalid",
    "topology_model_not_found",
    "topology_registry_unavailable",
    "topology_response_budget_exceeded",
    "topology_stored_metadata_invalid",
]
