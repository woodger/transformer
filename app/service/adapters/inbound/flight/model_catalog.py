from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from app.contracts.checkpoint.v7 import CHECKPOINT_FORMAT, validate_checkpoint_document
from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v2 import (
    CONTRACT_NAME,
    CONTRACT_REVISION,
    validate_catalog_document,
)
from app.contracts.semantic.v2 import ModelContract
from app.service.application.ports.model_catalog import (
    CatalogArtifactVerificationError,
)
from app.service.application.services.model_contract import (
    model_initialization,
    verify_model_integrity,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.service.domain.records import PublishedModelRecord

_CHECKPOINT_METADATA_FIELDS = (
    "format",
    "serviceVersion",
    "generation",
    "jobId",
    "dataContract",
    "modelContract",
    "semanticDigests",
    "trainingConfig",
    "diagnostics",
    "selection",
    "initialization",
    "jobConfigSha256",
    "manifestSha256",
    "progress",
)


class CatalogModelMetadataVerifier:
    """Validate the complete catalog projection before artifact I/O."""

    def verify(self, model: PublishedModelRecord) -> None:
        model_detail(model)


def model_summary(model: PublishedModelRecord) -> JsonObject:
    try:
        verify_model_integrity(model)
        contract = ModelContract.from_document(model.model_contract)
        initialization = model_initialization(model)
        metadata = model.metadata
        producing_run_id = metadata["jobId"]
        checkpoint = metadata["checkpoint"]
        if (
            metadata.get("modelRef") != model.model_ref
            or metadata.get("label") != model.label
            or not isinstance(producing_run_id, str)
            or (
                model.producing_job_id is not None
                and producing_run_id != model.producing_job_id
            )
            or checkpoint
            != {
                "format": CHECKPOINT_FORMAT,
                "sha256": model.sha256,
                "bytes": model.byte_count,
            }
        ):
            raise ValueError("published model projection is inconsistent")
        initialization_summary: JsonObject = {
            "source": initialization["source"]
        }
        if initialization["source"] == "publishedModel":
            initialization_summary["parentModelRef"] = initialization[
                "parentModelRef"
            ]
        summary: JsonObject = {
            "modelRef": model.model_ref,
            "label": model.label,
            "generation": model.generation,
            "createdAt": _timestamp(model.created_at),
            "semanticDigests": dict(model.semantic_digests),
            "modelConfig": dict(contract.model_config),
            "targetIdentities": list(contract.target_identities),
            "initialization": initialization_summary,
            "producingRunId": producing_run_id,
            "checkpoint": dict(cast(JsonObject, checkpoint)),
        }
        validate_catalog_document(
            {
                "contract": CONTRACT_NAME,
                "revision": CONTRACT_REVISION,
                "requestId": "00000000-0000-4000-8000-000000000000",
                "models": [summary],
                "nextCursor": None,
                "cursorExpiresAt": None,
            },
            "list-result",
        )
        return summary
    except ServiceError as exc:
        raise stored_metadata_invalid(model.model_ref, "") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise stored_metadata_invalid(model.model_ref, "") from exc


def model_detail(model: PublishedModelRecord) -> JsonObject:
    summary = model_summary(model)
    try:
        metadata = model.metadata
        checkpoint_metadata: JsonObject = {
            name: metadata[name] for name in _CHECKPOINT_METADATA_FIELDS
        }
        validate_checkpoint_document(
            checkpoint_metadata,
            "checkpoint-metadata",
        )
        progress = cast(JsonObject, checkpoint_metadata["progress"])
        if progress.get("trainingComplete") is not True:
            raise ValueError("published checkpoint is not terminal")
        selection = cast(JsonObject, checkpoint_metadata["selection"])
        if (
            selection.get("modelContractSha256")
            != model.semantic_digests.get("modelContractSha256")
        ):
            raise ValueError("selection model contract differs")
        detail: JsonObject = {
            "summary": summary,
            "dataContract": dict(model.data_contract),
            "modelContract": dict(model.model_contract),
            "trainingConfig": dict(
                cast(JsonObject, metadata["trainingConfig"])
            ),
            "diagnostics": dict(cast(JsonObject, metadata["diagnostics"])),
            "selection": dict(selection),
            "progress": dict(progress),
            "initialization": dict(
                cast(JsonObject, checkpoint_metadata["initialization"])
            ),
            "jobConfigSha256": cast(str, metadata["jobConfigSha256"]),
        }
        validate_catalog_document(
            {
                "contract": CONTRACT_NAME,
                "revision": CONTRACT_REVISION,
                "requestId": "00000000-0000-4000-8000-000000000000",
                "model": detail,
            },
            "detail-result",
        )
        return detail
    except ServiceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise stored_metadata_invalid(model.model_ref, "") from exc


def catalog_error(
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
    validate_catalog_document(detail, "error-detail")
    return ServiceError(code, message, detail=detail)


def invalid_catalog_query(message: str, path: str) -> ServiceError:
    return catalog_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_CATALOG_QUERY",
        message,
        path=path,
    )


def invalid_catalog_cursor() -> ServiceError:
    return catalog_error(
        ErrorCode.INVALID_ARGUMENT,
        "INVALID_CATALOG_CURSOR",
        "model catalog cursor is invalid",
        path="/cursor",
    )


def expired_catalog_cursor() -> ServiceError:
    return catalog_error(
        ErrorCode.FAILED_PRECONDITION,
        "CATALOG_CURSOR_EXPIRED",
        "model catalog cursor has expired",
        path="/cursor",
        restartRequired=True,
    )


def unavailable_catalog_revision(revision: int) -> ServiceError:
    return catalog_error(
        ErrorCode.FAILED_PRECONDITION,
        "CATALOG_QUERY_REVISION_UNAVAILABLE",
        "model catalog query revision is unavailable",
        requestedRevision=revision,
    )


def stored_metadata_invalid(model_ref: str, path: str) -> ServiceError:
    return catalog_error(
        ErrorCode.MODEL_CORRUPT,
        "STORED_MODEL_METADATA_INVALID",
        "stored model metadata is invalid",
        modelRef=model_ref,
        path=path,
    )


def model_not_found(model_ref: str) -> ServiceError:
    return catalog_error(
        ErrorCode.NOT_FOUND,
        "MODEL_NOT_FOUND",
        "model generation was not found",
        modelRef=model_ref,
    )


def registry_unavailable() -> ServiceError:
    return catalog_error(
        ErrorCode.UNAVAILABLE,
        "MODEL_REGISTRY_UNAVAILABLE",
        "model registry is temporarily unavailable",
    )


def artifact_error(error: CatalogArtifactVerificationError) -> ServiceError:
    code = (
        ErrorCode.RESOURCE_EXHAUSTED
        if error.reason == "CHECKPOINT_VERIFICATION_BUDGET_EXCEEDED"
        else ErrorCode.MODEL_CORRUPT
    )
    messages = {
        "CHECKPOINT_VERIFICATION_BUDGET_EXCEEDED": (
            "model checkpoint exceeds the verification budget"
        ),
        "STORED_MODEL_METADATA_INVALID": "stored model metadata is invalid",
        "MODEL_CHECKPOINT_UNAVAILABLE": "model checkpoint is unavailable",
        "MODEL_CHECKPOINT_SIZE_MISMATCH": (
            "model checkpoint size does not match the registry"
        ),
        "MODEL_CHECKPOINT_DIGEST_MISMATCH": (
            "model checkpoint digest does not match the registry"
        ),
    }
    message = messages.get(error.reason)
    if message is None:
        raise RuntimeError("unknown catalog artifact failure") from error
    return catalog_error(code, error.reason, message, **error.fields)


def _timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "CatalogModelMetadataVerifier",
    "artifact_error",
    "catalog_error",
    "expired_catalog_cursor",
    "invalid_catalog_cursor",
    "invalid_catalog_query",
    "model_detail",
    "model_not_found",
    "model_summary",
    "registry_unavailable",
    "stored_metadata_invalid",
    "unavailable_catalog_revision",
]
