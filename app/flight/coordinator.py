from __future__ import annotations

from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.worker_probe.device_inventory import (
    CudaDeviceInventory,
    static_cuda_inventory,
)
from app.service.bootstrap.job_control import build_job_coordinator


class JobCoordinator:
    """Build the current coordinator through the stable ``app.flight`` API."""

    def __new__(
        cls,
        config,
        ledger,
        spool,
        *,
        cuda_available=None,
        device_inventory: CudaDeviceInventory | None = None,
        recovery_store=None,
        metrics=None,
        logger=None,
        cancel_notifier=None,
        queue_notifier=None,
    ):
        if device_inventory is None:
            device_inventory = (
                static_cuda_inventory(cuda_available)
                if cuda_available is not None
                else CudaDeviceInventory().initialize()
            )
        return build_job_coordinator(
            config,
            ledger,
            spool,
            recovery_store or spool,
            device_inventory=device_inventory,
            metrics=metrics or OperationalMetrics(),
            logger=logger or JsonLogger(),
            cancel_notifier=cancel_notifier,
            queue_notifier=queue_notifier,
        )


__all__ = ["JobCoordinator"]
