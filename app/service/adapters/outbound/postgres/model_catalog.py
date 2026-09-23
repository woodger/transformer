from __future__ import annotations

import secrets
import threading
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from app.service.adapters.outbound.postgres.ledger.support import advisory_lock
from app.service.adapters.outbound.postgres.mapping import published_model_record
from app.service.adapters.outbound.postgres.models import (
    PublishedModel,
    RuntimeState,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.application.messages.model_catalog import (
    CatalogModelRecord,
    StoredCatalogPage,
)
from app.service.application.ports.model_catalog import (
    ModelCatalogStoreUnavailable,
)
from app.service.domain.model import ModelLifecycleState

_CURSOR_KEY_STATE = "model_catalog_cursor_hmac_v3"


class PostgresModelCatalogStore:
    """Owner-scoped read model for the public model catalog."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._cursor_key: bytes | None = None
        self._cursor_key_lock = threading.Lock()

    def cursor_signing_key(self) -> bytes:
        try:
            with self._cursor_key_lock:
                if self._cursor_key is not None:
                    return self._cursor_key
                candidate = secrets.token_hex(32)
                now = datetime.now(UTC)
                with self._database.transaction() as session:
                    session.execute(
                        insert(RuntimeState)
                        .values(
                            key=_CURSOR_KEY_STATE,
                            value=candidate,
                            updated_at=now,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[RuntimeState.key]
                        )
                    )
                    value = session.scalar(
                        select(RuntimeState.value).where(
                            RuntimeState.key == _CURSOR_KEY_STATE
                        )
                    )
                if not isinstance(value, str):
                    raise ModelCatalogStoreUnavailable(
                        "model catalog cursor key is unavailable"
                    )
                try:
                    key = bytes.fromhex(value)
                except ValueError as exc:
                    raise ModelCatalogStoreUnavailable(
                        "model catalog cursor key is invalid"
                    ) from exc
                if len(key) != 32:
                    raise ModelCatalogStoreUnavailable(
                        "model catalog cursor key is invalid"
                    )
                self._cursor_key = key
                return key
        except SQLAlchemyError as exc:
            raise ModelCatalogStoreUnavailable(
                "model catalog registry is unavailable"
            ) from exc

    def first_page(
        self,
        owner_subject: str,
        *,
        limit: int,
    ) -> StoredCatalogPage:
        try:
            with self._database.transaction() as session:
                # Публикация получает ту же блокировку владельца перед выделением
                # порядкового номера каталога. Ни один незафиксированный номер,
                # не превышающий эту верхнюю границу, не станет видимым позже.
                advisory_lock(
                    session,
                    "model-catalog-publication",
                    owner_subject,
                )
                high_water = session.scalar(
                    select(func.max(PublishedModel.catalog_ordinal)).where(
                        PublishedModel.owner_subject == owner_subject,
                        PublishedModel.lifecycle_state
                        == ModelLifecycleState.AVAILABLE.value,
                    )
                )
                return _page(
                    session.scalars(
                        _base_query(owner_subject, int(high_water or 0))
                        .limit(limit + 1)
                    ).all(),
                    high_water_ordinal=int(high_water or 0),
                    limit=limit,
                )
        except SQLAlchemyError as exc:
            raise ModelCatalogStoreUnavailable(
                "model catalog registry is unavailable"
            ) from exc

    def continuation_page(
        self,
        owner_subject: str,
        *,
        high_water_ordinal: int,
        after_created_at: datetime,
        after_model_ref: str,
        limit: int,
    ) -> StoredCatalogPage:
        try:
            with self._database.session() as session:
                statement = _base_query(
                    owner_subject, high_water_ordinal
                ).where(
                    or_(
                        PublishedModel.created_at < after_created_at,
                        and_(
                            PublishedModel.created_at == after_created_at,
                            PublishedModel.model_ref > after_model_ref,
                        ),
                    )
                )
                return _page(
                    session.scalars(statement.limit(limit + 1)).all(),
                    high_water_ordinal=high_water_ordinal,
                    limit=limit,
                )
        except SQLAlchemyError as exc:
            raise ModelCatalogStoreUnavailable(
                "model catalog registry is unavailable"
            ) from exc

    def get_model(
        self,
        owner_subject: str,
        model_ref: str,
    ) -> CatalogModelRecord | None:
        try:
            with self._database.session() as session:
                row = session.scalar(
                    select(PublishedModel).where(
                        PublishedModel.owner_subject == owner_subject,
                        PublishedModel.model_ref == model_ref,
                        PublishedModel.lifecycle_state
                        == ModelLifecycleState.AVAILABLE.value,
                    )
                )
                return None if row is None else _record(row)
        except SQLAlchemyError as exc:
            raise ModelCatalogStoreUnavailable(
                "model catalog registry is unavailable"
            ) from exc


def _base_query(
    owner_subject: str,
    high_water_ordinal: int,
) -> Select[tuple[PublishedModel]]:
    return (
        select(PublishedModel)
        .where(
            PublishedModel.owner_subject == owner_subject,
            PublishedModel.lifecycle_state
            == ModelLifecycleState.AVAILABLE.value,
            PublishedModel.catalog_ordinal <= high_water_ordinal,
        )
        .order_by(
            PublishedModel.created_at.desc(),
            PublishedModel.model_ref.asc(),
        )
    )


def _page(
    rows: Sequence[PublishedModel],
    *,
    high_water_ordinal: int,
    limit: int,
) -> StoredCatalogPage:
    return StoredCatalogPage(
        high_water_ordinal=high_water_ordinal,
        items=tuple(_record(row) for row in rows[:limit]),
        has_more=len(rows) > limit,
    )


def _record(row: PublishedModel) -> CatalogModelRecord:
    model = published_model_record(row)
    if model is None:
        raise AssertionError("published model mapping unexpectedly returned null")
    return CatalogModelRecord(
        model=model,
        catalog_ordinal=row.catalog_ordinal,
        created_at=row.created_at,
    )


__all__ = ["PostgresModelCatalogStore"]
