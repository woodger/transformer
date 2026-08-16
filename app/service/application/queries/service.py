from __future__ import annotations

from dataclasses import dataclass

from app.service.application.messages.jobs import ServiceLimits
from app.service.application.ports.devices import (
    WorkerCapabilities,
    WorkerCapabilitySnapshot,
)
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.operations import (
    DiskUsage,
    ServiceHealthRepository,
    StorageUsageReader,
)
from app.service.domain.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class StorageUsage:
    total: int
    used: int
    free: int


@dataclass(frozen=True, slots=True)
class ServiceCapabilities:
    device_inventory: WorkerCapabilitySnapshot
    cpu_capacity: int
    limits: ServiceLimits


@dataclass(frozen=True, slots=True)
class ServiceHealth:
    ready: bool
    draining: bool
    ledger_available: bool
    device_inventory: WorkerCapabilitySnapshot
    runtime_storage: StorageUsage
    recovery_storage: StorageUsage
    metrics: JsonObject


class ServiceAvailability:
    def __init__(self) -> None:
        self._draining = False

    @property
    def draining(self) -> bool:
        return self._draining

    def set_draining(self, value: bool = True) -> None:
        self._draining = bool(value)


class ServiceStatusQuery:
    def __init__(
        self,
        *,
        health_repository: ServiceHealthRepository,
        runtime_storage: StorageUsageReader,
        recovery_storage: StorageUsageReader,
        device_inventory: WorkerCapabilities,
        availability: ServiceAvailability,
        cpu_capacity: int,
        limits: ServiceLimits,
        metrics: OperationalMetricSink,
        logger: EventLogger,
    ) -> None:
        self.health_repository = health_repository
        self.runtime_storage = runtime_storage
        self.recovery_storage = recovery_storage
        self.device_inventory = device_inventory
        self.availability = availability
        self.cpu_capacity = cpu_capacity
        self.limits = limits
        self.metrics = metrics
        self.logger = logger

    def capabilities(self) -> ServiceCapabilities:
        inventory = self.device_inventory.snapshot()
        self.metrics.set("cudaAvailable", inventory.cuda_capacity > 0)
        return ServiceCapabilities(
            device_inventory=inventory,
            cpu_capacity=self.cpu_capacity,
            limits=self.limits,
        )

    def health(self) -> ServiceHealth:
        runtime_usage = _usage(self.runtime_storage.disk_usage())
        recovery_usage = _usage(self.recovery_storage.disk_usage())
        try:
            ledger_ready = bool(self.health_repository.healthcheck())
        except Exception as exc:
            ledger_ready = False
            self.logger.event(
                "flight.ledger.health_failed",
                errorType=type(exc).__name__,
            )
        ready = not self.availability.draining and ledger_ready
        inventory = self.device_inventory.snapshot()
        self.metrics.set("cudaAvailable", inventory.cuda_capacity > 0)
        self.metrics.set("ready", ready)
        self.metrics.set("diskTotalBytes", runtime_usage.total)
        self.metrics.set("diskUsedBytes", runtime_usage.used)
        self.metrics.set("diskFreeBytes", runtime_usage.free)
        self.metrics.set("recoveryDiskTotalBytes", recovery_usage.total)
        self.metrics.set("recoveryDiskUsedBytes", recovery_usage.used)
        self.metrics.set("recoveryDiskFreeBytes", recovery_usage.free)
        return ServiceHealth(
            ready=ready,
            draining=self.availability.draining,
            ledger_available=ledger_ready,
            device_inventory=inventory,
            runtime_storage=runtime_usage,
            recovery_storage=recovery_usage,
            metrics=self.metrics.snapshot(),
        )


def _usage(value: DiskUsage) -> StorageUsage:
    return StorageUsage(value.total, value.used, value.free)


__all__ = [
    "ServiceAvailability",
    "ServiceCapabilities",
    "ServiceHealth",
    "ServiceStatusQuery",
    "StorageUsage",
]
