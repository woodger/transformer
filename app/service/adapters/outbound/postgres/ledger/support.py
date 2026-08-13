from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Protocol, cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.contracts.json_types import JsonObject
from app.service.adapters.outbound.postgres.session import Database

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

RowMapping = dict[str, object]


class _MappedColumn(Protocol):
    name: str


class _MappedAttribute(Protocol):
    key: str
    columns: Sequence[_MappedColumn]


class _Mapper(Protocol):
    column_attrs: Iterable[_MappedAttribute]


class _MappedRecord(Protocol):
    __mapper__: _Mapper


class LedgerSessions:
    """Concrete PostgreSQL session scopes shared by ledger use-case slices."""

    def __init__(self, database: Database) -> None:
        self.database = database

    @contextmanager
    def read(self, connection: Session | None) -> Generator[Session]:
        if connection is not None:
            yield connection
            return
        with self.database.session() as session:
            yield session

    @contextmanager
    def write(self, connection: Session | None) -> Generator[Session]:
        if connection is not None:
            yield connection
            return
        with self.database.transaction() as session:
            yield session


def timestamp(value: datetime | None) -> float | None:
    return None if value is None else value.timestamp()


def decode(record: object) -> RowMapping:
    mapped_record = cast(_MappedRecord, record)
    result: RowMapping = {}
    for attribute in mapped_record.__mapper__.column_attrs:
        column = attribute.columns[0]
        value = cast(object, getattr(record, attribute.key))
        if isinstance(value, datetime):
            value = value.timestamp()
        result[column.name] = value
    return result


def decode_optional(record: object | None) -> RowMapping | None:
    return None if record is None else decode(record)


def json_value(value: object) -> JsonObject:
    if is_dataclass(value):
        if isinstance(value, type):
            raise TypeError("JSON value must be a dataclass instance")
        value = asdict(value)
    document = json.loads(json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ))
    if not isinstance(document, dict):
        raise TypeError("ledger JSON document must be an object")
    return cast(JsonObject, document)


def canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} must be a canonical UUID")
    return str(parsed)


def validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError("artifact path must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("artifact path must not contain path traversal")
    return value


def nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def now(value: float | None) -> datetime:
    return datetime.now(UTC) if value is None else at(value)


def at(value: float) -> datetime:
    return datetime.fromtimestamp(float(value), UTC)


def advisory_lock(session: Session, *parts: str) -> None:
    value = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    key = int.from_bytes(value[:8], "big", signed=True)
    session.scalar(select(func.pg_advisory_xact_lock(key)))
