from __future__ import annotations

import hashlib
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import rfc8785

from app.service.domain.json_types import JsonValue

TelemetrySnapshotOperation = Literal[
    "report",
    "gradientInteractions",
    "targetHeadDiagnostics",
]


class TrainingTelemetrySnapshotCapacityExhausted(RuntimeError):
    def __init__(self, operation: TelemetrySnapshotOperation) -> None:
        super().__init__(operation)
        self.operation = operation


@dataclass(frozen=True, slots=True)
class TrainingTelemetrySnapshotUsage:
    count: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class _SnapshotEntry:
    value: object
    content_sha256: str
    byte_count: int
    expires_at: datetime


class TrainingTelemetrySnapshotStore:
    """Удерживает точные cursor-проекции в одном атомарно ограниченном пуле."""

    def __init__(
        self,
        *,
        max_count: int,
        max_total_bytes: int,
        max_snapshot_bytes: int,
        boot_identity: str | None = None,
    ) -> None:
        if max_count < 1:
            raise ValueError("snapshot count limit must be positive")
        if max_total_bytes < 1:
            raise ValueError("snapshot total byte limit must be positive")
        if not 1 <= max_snapshot_bytes <= max_total_bytes:
            raise ValueError(
                "snapshot byte limit must be positive and not exceed total"
            )
        identity = (
            secrets.token_hex(16)
            if boot_identity is None
            else boot_identity
        )
        if (
            len(identity) != 32
            or identity != identity.lower()
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise ValueError("snapshot boot identity must be 128-bit lowercase hex")
        self._boot_identity = identity
        self._max_count = max_count
        self._max_total_bytes = max_total_bytes
        self._max_snapshot_bytes = max_snapshot_bytes
        self._entries: dict[tuple[str, ...], _SnapshotEntry] = {}
        self._total_bytes = 0
        self._lock = threading.Lock()

    @property
    def boot_identity(self) -> str:
        return self._boot_identity

    def get(
        self,
        key: tuple[str, ...],
        *,
        now: datetime,
    ) -> object | None:
        with self._lock:
            self._evict_expired(now)
            entry = self._entries.get(key)
            return None if entry is None else entry.value

    def admit(
        self,
        key: tuple[str, ...],
        value: object,
        projection: JsonValue,
        *,
        operation: TelemetrySnapshotOperation,
        expires_at: datetime,
        now: datetime,
    ) -> object:
        canonical = rfc8785.dumps(projection)
        byte_count = len(canonical)
        content_sha256 = hashlib.sha256(canonical).hexdigest()
        with self._lock:
            self._evict_expired(now)
            existing = self._entries.get(key)
            if existing is not None:
                if (
                    existing.byte_count != byte_count
                    or existing.content_sha256 != content_sha256
                ):
                    raise ValueError(
                        "retained telemetry snapshot identity is inconsistent"
                    )
                if existing.expires_at < expires_at:
                    self._entries[key] = _SnapshotEntry(
                        existing.value,
                        existing.content_sha256,
                        existing.byte_count,
                        expires_at,
                    )
                return existing.value
            # Не вытесняем ещё действующую запись: выданный курсор должен хранить
            # эту точную проекцию до истечения срока.
            if (
                byte_count > self._max_snapshot_bytes
                or len(self._entries) >= self._max_count
                or self._total_bytes + byte_count > self._max_total_bytes
            ):
                raise TrainingTelemetrySnapshotCapacityExhausted(operation)
            self._entries[key] = _SnapshotEntry(
                value,
                content_sha256,
                byte_count,
                expires_at,
            )
            self._total_bytes += byte_count
            return value

    def usage(self, *, now: datetime) -> TrainingTelemetrySnapshotUsage:
        with self._lock:
            self._evict_expired(now)
            return TrainingTelemetrySnapshotUsage(
                count=len(self._entries),
                byte_count=self._total_bytes,
            )

    def _evict_expired(self, now: datetime) -> None:
        expired = [
            key for key, entry in self._entries.items() if now >= entry.expires_at
        ]
        for key in expired:
            self._total_bytes -= self._entries.pop(key).byte_count


__all__ = [
    "TelemetrySnapshotOperation",
    "TrainingTelemetrySnapshotCapacityExhausted",
    "TrainingTelemetrySnapshotStore",
    "TrainingTelemetrySnapshotUsage",
]
