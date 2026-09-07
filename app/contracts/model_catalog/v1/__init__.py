"""Current model catalog query revision 1 contract."""
from app.contracts.model_catalog.v1.capabilities import catalog_capabilities
from app.contracts.model_catalog.v1.codec import (
    ModelCatalogContractError,
    validate_catalog_document,
)
from app.contracts.model_catalog.v1.constants import (
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
    "catalog_capabilities",
    "validate_catalog_document",
]
