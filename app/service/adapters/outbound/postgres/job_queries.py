from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.mapping import (
    row_boolean,
    row_integer,
    row_optional_float,
    row_optional_integer,
    row_string,
)
from app.service.application.job_models import StoredInputPage, StoredOutputPage
from app.service.domain.records import (
    InputRecord,
    OutputRecord,
    PublishedModelRecord,
    StatusSnapshot,
)


class PostgresJobQueryStore:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def get_status_snapshot_record(
        self,
        job_id: str,
        owner_subject: str,
    ) -> StatusSnapshot:
        return self.ledger.get_status_snapshot_record(job_id, owner_subject)

    def is_job_retired(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> bool:
        identity = self.ledger.get_job_identity(
            job_id,
            owner_subject=owner_subject,
        )
        return identity is not None and identity["retired_at"] is not None

    def list_inputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        after_revision: int,
        snapshot_revision: int | None,
        cursor: int | None,
        limit: int,
    ) -> StoredInputPage:
        page = self.ledger.list_inputs_page(
            job_id,
            owner_subject,
            after_revision=after_revision,
            snapshot_revision=snapshot_revision,
            cursor=cursor,
            limit=limit,
        )
        return StoredInputPage(
            after_revision=row_integer(page, "after_revision"),
            snapshot_revision=row_integer(page, "snapshot_revision"),
            cursor=row_integer(page, "cursor"),
            items=tuple(
                _input_record(item)
                for item in _mapping_items(page, "items")
            ),
            next_cursor=row_optional_integer(page, "next_cursor"),
            has_more=row_boolean(page, "has_more"),
        )

    def list_outputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        cursor: int | None,
        limit: int,
    ) -> StoredOutputPage:
        page = self.ledger.list_outputs_page(
            job_id,
            owner_subject,
            cursor=cursor,
            limit=limit,
        )
        return StoredOutputPage(
            cursor=row_optional_integer(page, "cursor"),
            items=tuple(
                _output_record(item)
                for item in _mapping_items(page, "items")
            ),
            next_cursor=row_optional_integer(page, "next_cursor"),
            has_more=row_boolean(page, "has_more"),
        )

    def get_published_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
    ) -> PublishedModelRecord | None:
        return self.ledger.get_published_model(
            model_ref,
            owner_subject=owner_subject,
        )

    def resolve_published_model_alias(
        self,
        owner_subject: str,
        label: str,
    ) -> PublishedModelRecord | None:
        return self.ledger.resolve_published_model_alias(
            owner_subject,
            label,
        )


def _input_record(item: Mapping[str, object]) -> InputRecord:
    return InputRecord(
        job_id=row_string(item, "job_id"),
        ordinal=row_integer(item, "ordinal"),
        payload_id=row_string(item, "payload_id"),
        commit_revision=row_integer(item, "commit_revision"),
        schema_id=row_string(item, "schema_id"),
        data_contract_sha256=row_string(item, "data_contract_sha256"),
        rows=row_integer(item, "rows"),
        batches=row_integer(item, "batches"),
        byte_count=row_integer(item, "bytes"),
        sha256=row_string(item, "sha256"),
        schema_fingerprint=row_string(item, "schema_fingerprint"),
        relative_path=row_string(item, "relative_path"),
        storage_class=row_string(item, "storage_class"),
        source_width=row_integer(item, "source_width"),
        feature_dim=row_integer(item, "feature_dim"),
        committed_at=_required_float(item, "committed_at"),
    )


def _output_record(item: Mapping[str, object]) -> OutputRecord:
    return OutputRecord(
        job_id=row_string(item, "job_id"),
        ordinal=row_integer(item, "ordinal"),
        rows=row_integer(item, "rows"),
        batches=row_integer(item, "batches"),
        byte_count=row_integer(item, "bytes"),
        sha256=row_string(item, "sha256"),
        schema_fingerprint=row_string(item, "schema_fingerprint"),
        relative_path=row_string(item, "relative_path"),
        published_at=_required_float(item, "published_at"),
    )


def _mapping_items(
    value: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...]:
    items = value.get(key)
    if not isinstance(items, list):
        raise ValueError(f"database field {key} must be a list")
    object_items = cast(list[object], items)
    if not all(isinstance(item, Mapping) for item in object_items):
        raise ValueError(f"database field {key} contains a non-object item")
    return tuple(
        cast(Mapping[str, object], item) for item in object_items
    )


def _required_float(value: Mapping[str, object], key: str) -> float:
    result = row_optional_float(value, key)
    if result is None:
        raise ValueError(f"database field {key} must be numeric")
    return result


__all__ = ["PostgresJobQueryStore"]
