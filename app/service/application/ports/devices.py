from __future__ import annotations

from typing import Protocol


class WorkerDevice(Protocol):
    device_id: str
    ordinal: int
    name: str


class WorkerCapabilitySnapshot(Protocol):
    devices: tuple[WorkerDevice, ...]
    runtime_version: str | None
    torch_version: str

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

    def schedulable_devices(self) -> tuple[WorkerDevice, ...]: ...

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
