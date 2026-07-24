from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from sqlalchemy import func, select
from sqlalchemy.orm import Session

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LedgerSessions:
    """Concrete PostgreSQL session scopes shared by ledger use-case slices."""

    def __init__(self, database):
        self.database = database

    @contextmanager
    def read(self, connection: Session | None) -> Iterator[Session]:
        if connection is not None:
            yield connection
            return
        with self.database.session() as session:
            yield session

    @contextmanager
    def write(self, connection: Session | None) -> Iterator[Session]:
        if connection is not None:
            yield connection
            return
        with self.database.transaction() as session:
            yield session


def timestamp(value: datetime | None) -> float | None:
    return None if value is None else value.timestamp()


def decode(record) -> dict | None:
    if record is None:
        return None
    result = {}
    for attribute in record.__mapper__.column_attrs:
        column = attribute.columns[0]
        value = getattr(record, attribute.key)
        if isinstance(value, datetime):
            value = value.timestamp()
        result[column.name] = value
    return result


def json_value(value):
    if value is None:
        return None
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} must be a canonical UUID")
    return str(parsed)


def validate_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError("artifact path must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("artifact path must not contain path traversal")


def nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def now(value: float | None) -> datetime:
    return datetime.now(UTC) if value is None else at(value)


def at(value: float) -> datetime:
    return datetime.fromtimestamp(float(value), UTC)


def advisory_lock(session: Session, *parts: str) -> None:
    value = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    key = int.from_bytes(value[:8], "big", signed=True)
    session.scalar(select(func.pg_advisory_xact_lock(key)))
