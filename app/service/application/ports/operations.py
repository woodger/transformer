from __future__ import annotations

from typing import Protocol


class ServiceHealthRepository(Protocol):
    def healthcheck(self) -> bool: ...


class DiskUsage(Protocol):
    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...

    @property
    def free(self) -> int: ...


class StorageUsageReader(Protocol):
    def disk_usage(self) -> DiskUsage: ...


__all__ = ["DiskUsage", "ServiceHealthRepository", "StorageUsageReader"]
