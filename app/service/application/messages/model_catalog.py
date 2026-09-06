from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.service.domain.records import PublishedModelRecord


@dataclass(frozen=True, slots=True)
class CatalogModelRecord:
    model: PublishedModelRecord
    catalog_ordinal: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredCatalogPage:
    high_water_ordinal: int
    items: tuple[CatalogModelRecord, ...]
    has_more: bool


@dataclass(frozen=True, slots=True)
class ListCatalogModelsQuery:
    owner_subject: str
    request_id: str
    page_size: int
    cursor: str | None


@dataclass(frozen=True, slots=True)
class GetCatalogModelQuery:
    owner_subject: str
    request_id: str
    model_ref: str


@dataclass(frozen=True, slots=True)
class CatalogModelsPage:
    request_id: str
    models: tuple[PublishedModelRecord, ...]
    next_cursor: str | None
    cursor_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class CatalogModelDetail:
    request_id: str
    model: PublishedModelRecord


__all__ = [
    "CatalogModelDetail",
    "CatalogModelRecord",
    "CatalogModelsPage",
    "GetCatalogModelQuery",
    "ListCatalogModelsQuery",
    "StoredCatalogPage",
]
