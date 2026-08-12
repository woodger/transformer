from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class WorkerDevice(Protocol):
    @property
    def device_id(self) -> str: ...

    @property
    def ordinal(self) -> int: ...

    @property
    def name(self) -> str: ...


class WorkerCapabilitySnapshot(Protocol):
    @property
    def devices(self) -> Sequence[WorkerDevice]: ...

    @property
    def runtime_version(self) -> str | None: ...

    @property
    def torch_version(self) -> str: ...

    @property
    def device_count(self) -> int: ...

    @property
    def cuda_capacity(self) -> int: ...

    @property
    def quarantined_count(self) -> int: ...


class WorkerCapabilities(Protocol):
    """Inspect worker runtime without importing its implementation."""

    def snapshot(self) -> WorkerCapabilitySnapshot: ...


class DeviceLeaseManager(WorkerCapabilities, Protocol):
    """Lease opaque execution devices and quarantine confirmed failures."""

    def schedulable_devices(self) -> Sequence[WorkerDevice]: ...

    def mark_busy(self, device_id: str) -> bool: ...

    def is_quarantined(self, device_id: str) -> bool: ...

    def release(self, device_id: str) -> None: ...

    def confirm_loss(self, device_id: str) -> bool: ...


__all__ = [
    "DeviceLeaseManager",
    "WorkerCapabilities",
    "WorkerCapabilitySnapshot",
    "WorkerDevice",
]
