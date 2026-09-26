from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from app.contracts.checkpoint.v12 import validate_checkpoint_document
from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v7 import validate_catalog_document
from app.contracts.semantic.v5 import ModelContract
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
    "predictionDefinition",
    "modelConfig",
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
    """Проверить полную проекцию каталога до ввода-вывода артефакта."""

    def verify(self, model: PublishedModelRecord) -> None:
        model_detail(model)


def model_summary(model: PublishedModelRecord) -> JsonObject:
    try:
        verify_model_integrity(model)
        contract = ModelContract.from_document(model.model_contract)
        initialization = model_initialization(model)
        metadata = model.metadata
        producing_run_id = metadata["jobId"]
        if (
            metadata.get("modelRef") != model.model_ref
            or metadata.get("label") != model.label
            or not isinstance(producing_run_id, str)
            or (
                model.producing_job_id is not None
                and producing_run_id != model.producing_job_id
            )
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
            "dataContractSha256": model.semantic_digests[
                "dataContractSha256"
            ],
            "modelDefinitionSha256": model.semantic_digests[
                "modelDefinitionSha256"
            ],
            "modelTuning": contract.model_tuning,
            "targetIdentities": list(contract.target_identities),
            "initialization": initialization_summary,
            "producingRunId": producing_run_id,
        }
        validate_catalog_document(
            {
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
        checkpoint_metadata = _checkpoint_metadata(metadata)
        progress = cast(JsonObject, checkpoint_metadata["progress"])
        if progress.get("trainingComplete") is not True:
            raise ValueError("published checkpoint is not terminal")
        selection = cast(JsonObject, checkpoint_metadata["selection"])
        if (
            selection.get("modelDefinitionSha256")
            != model.semantic_digests.get("modelDefinitionSha256")
        ):
            raise ValueError("selection model contract differs")
        selection_public = {
            key: value
            for key, value in selection.items()
            if key != "modelDefinitionSha256"
        }
        detail: JsonObject = {
            "summary": summary,
            "dataDefinition": {
                "dataContractSha256": model.semantic_digests[
                    "dataContractSha256"
                ],
                "tensorGeometry": {
                    "seqLen": model.data_contract["seqLen"],
                    "featureDim": model.data_contract["featureDim"],
                },
            },
            "modelContract": dict(model.model_contract),
            "predictionDefinition": dict(
                cast(JsonObject, checkpoint_metadata["predictionDefinition"])
            ),
            "semanticDigests": dict(model.semantic_digests),
            "trainingConfig": dict(
                cast(JsonObject, metadata["trainingConfig"])
            ),
            "diagnostics": _catalog_diagnostics(metadata["diagnostics"]),
            "selection": selection_public,
            "progress": dict(progress),
            "initialization": dict(cast(JsonObject, summary["initialization"])),
        }
        validate_catalog_document(
            {
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
    if error.reason == "CHECKPOINT_VERIFICATION_BUDGET_EXCEEDED":
        return catalog_error(
            ErrorCode.RESOURCE_EXHAUSTED,
            "MODEL_VERIFICATION_UNAVAILABLE",
            "model verification is temporarily unavailable",
        )
    if error.reason not in {
        "STORED_MODEL_METADATA_INVALID",
        "MODEL_CHECKPOINT_UNAVAILABLE",
        "MODEL_CHECKPOINT_SIZE_MISMATCH",
        "MODEL_CHECKPOINT_DIGEST_MISMATCH",
    }:
        raise RuntimeError("unknown catalog artifact failure") from error
    return catalog_error(
        ErrorCode.MODEL_CORRUPT,
        "MODEL_CHECKPOINT_INVALID",
        "model checkpoint is invalid",
    )


def _timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _checkpoint_metadata(metadata: JsonObject) -> JsonObject:
    checkpoint_metadata: JsonObject = {
        name: metadata[name] for name in _CHECKPOINT_METADATA_FIELDS
    }
    validate_checkpoint_document(
        checkpoint_metadata,
        "checkpoint-metadata",
    )
    return checkpoint_metadata


def _catalog_diagnostics(value: object) -> JsonObject:
    diagnostics = cast(JsonObject, value)
    if diagnostics.get("schemaVersion") == 1:
        return {
            "schemaVersion": 3,
            "gradientInteractions": diagnostics["gradientInteractions"],
            "targetHead": None,
            "encoderLayerDiagnostics": None,
        }
    if diagnostics.get("schemaVersion") == 2:
        return {
            "schemaVersion": 3,
            "gradientInteractions": diagnostics["gradientInteractions"],
            "targetHead": diagnostics["targetHead"],
            "encoderLayerDiagnostics": None,
        }
    return dict(diagnostics)


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
]
