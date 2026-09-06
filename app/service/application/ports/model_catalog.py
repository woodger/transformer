from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.service.application.messages.model_catalog import (
    CatalogModelRecord,
    StoredCatalogPage,
)
from app.service.domain.records import PublishedModelRecord


class ModelCatalogStoreUnavailable(RuntimeError):
    pass


class CatalogModelNotFound(LookupError):
    def __init__(self, model_ref: str) -> None:
        super().__init__(model_ref)
        self.model_ref = model_ref


class CatalogArtifactVerificationError(RuntimeError):
    def __init__(self, reason: str, **fields: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fields = fields


class ModelCatalogStore(Protocol):
    def cursor_signing_key(self) -> bytes: ...

    def first_page(
        self,
        owner_subject: str,
        *,
        limit: int,
    ) -> StoredCatalogPage: ...

    def continuation_page(
        self,
        owner_subject: str,
        *,
        high_water_ordinal: int,
        after_created_at: datetime,
        after_model_ref: str,
        limit: int,
    ) -> StoredCatalogPage: ...

    def get_model(
        self,
        owner_subject: str,
        model_ref: str,
    ) -> CatalogModelRecord | None: ...


class CatalogArtifactVerifier(Protocol):
    def verify(self, model: PublishedModelRecord) -> None: ...


__all__ = [
    "CatalogArtifactVerificationError",
    "CatalogArtifactVerifier",
    "CatalogModelNotFound",
    "ModelCatalogStore",
    "ModelCatalogStoreUnavailable",
]
