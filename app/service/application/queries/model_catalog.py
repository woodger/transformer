from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from app.service.application.messages.model_catalog import (
    CatalogModelDetail,
    CatalogModelsPage,
    GetCatalogModelQuery,
    ListCatalogModelsQuery,
)
from app.service.application.ports.model_catalog import (
    CatalogArtifactVerificationError,
    CatalogArtifactVerifier,
    CatalogModelNotFound,
    ModelCatalogStore,
)
from app.service.application.services.model_catalog_cursor import (
    CatalogCursor,
    CatalogCursorCodec,
)


class ListCatalogModels:
    def __init__(
        self,
        store: ModelCatalogStore,
        *,
        cursor_ttl_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._clock = clock
        self._cursor_ttl_seconds = cursor_ttl_seconds
        self._codec: CatalogCursorCodec | None = None
        self._codec_lock = threading.Lock()

    def execute(self, query: ListCatalogModelsQuery) -> CatalogModelsPage:
        now = self._clock().astimezone(UTC)
        if query.cursor is None:
            page = self._store.first_page(
                query.owner_subject,
                limit=query.page_size,
            )
            expires_at = now + timedelta(seconds=self._cursor_ttl_seconds)
        else:
            cursor = self._cursor_codec().decode(
                query.owner_subject,
                query.cursor,
                page_size=query.page_size,
                now=now,
            )
            page = self._store.continuation_page(
                query.owner_subject,
                high_water_ordinal=cursor.high_water_ordinal,
                after_created_at=cursor.after_created_at,
                after_model_ref=cursor.after_model_ref,
                limit=query.page_size,
            )
            expires_at = cursor.expires_at

        models = tuple(item.model for item in page.items)
        if not page.has_more:
            return CatalogModelsPage(query.request_id, models, None, None)
        if not page.items:
            raise RuntimeError("catalog page cannot continue without a sort key")
        last = page.items[-1]
        next_cursor = self._cursor_codec().encode(
            query.owner_subject,
            CatalogCursor(
                page_size=query.page_size,
                high_water_ordinal=page.high_water_ordinal,
                after_created_at=last.created_at,
                after_model_ref=last.model.model_ref,
                expires_at=expires_at,
            ),
        )
        return CatalogModelsPage(
            query.request_id,
            models,
            next_cursor,
            expires_at,
        )

    def _cursor_codec(self) -> CatalogCursorCodec:
        with self._codec_lock:
            if self._codec is None:
                self._codec = CatalogCursorCodec(
                    self._store.cursor_signing_key()
                )
            return self._codec


class GetCatalogModel:
    def __init__(
        self,
        store: ModelCatalogStore,
        *,
        artifact_verifier: CatalogArtifactVerifier,
    ) -> None:
        self._store = store
        self._artifact_verifier = artifact_verifier

    def execute(self, query: GetCatalogModelQuery) -> CatalogModelDetail:
        entry = self._store.get_model(
            query.owner_subject,
            query.model_ref,
        )
        if entry is None:
            raise CatalogModelNotFound(query.model_ref)
        try:
            self._artifact_verifier.verify(entry.model)
        except CatalogArtifactVerificationError:
            # Deletion may win after the initial registry read. Preserve the
            # security-equivalent not-found outcome instead of exposing a
            # transient filesystem symptom for a no-longer-visible model.
            if self._store.get_model(
                query.owner_subject,
                query.model_ref,
            ) is None:
                raise CatalogModelNotFound(query.model_ref) from None
            raise
        return CatalogModelDetail(query.request_id, entry.model)


__all__ = ["GetCatalogModel", "ListCatalogModels"]
