"""Подготовленный контракт Model Catalog Query v6 в области владельца."""

from app.contracts.model_catalog.v6.codec import (
    ModelCatalogContractError,
    validate_catalog_document,
)
from app.contracts.model_catalog.v6.constants import (
    CONTRACT_NAME,
    CONTRACT_REVISION,
    CURSOR_TTL_SECONDS,
    DETAIL_ACTION,
    LIST_ACTION,
    MAX_CHECKPOINT_VERIFICATION_BYTES,
    MAX_PAGE_SIZE,
    MAX_RESPONSE_BYTES,
)

__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_REVISION",
    "CURSOR_TTL_SECONDS",
    "DETAIL_ACTION",
    "LIST_ACTION",
    "MAX_CHECKPOINT_VERIFICATION_BYTES",
    "MAX_PAGE_SIZE",
    "MAX_RESPONSE_BYTES",
    "ModelCatalogContractError",
    "validate_catalog_document",
]
