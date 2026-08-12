from __future__ import annotations

from typing import Protocol


class ServiceHealthRepository(Protocol):
    def healthcheck(self) -> bool: ...


class StorageUsageReader(Protocol):
    def disk_usage(self): ...


__all__ = ["ServiceHealthRepository", "StorageUsageReader"]
