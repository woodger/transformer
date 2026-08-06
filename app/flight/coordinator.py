from __future__ import annotations

from app.service.adapters.inbound.flight.coordinator import (
    JobCoordinator as FlightJobCoordinator,
)
from app.service.adapters.outbound.worker_probe.device_inventory import (
    CudaDeviceInventory,
    static_cuda_inventory,
)


class JobCoordinator(FlightJobCoordinator):
    """Compatibility facade which supplies the historical CUDA probe seam."""

    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        cuda_available=None,
        device_inventory: CudaDeviceInventory | None = None,
        **kwargs,
    ):
        if device_inventory is None:
            device_inventory = (
                static_cuda_inventory(cuda_available)
                if cuda_available is not None
                else CudaDeviceInventory().initialize()
            )
        super().__init__(
            config,
            ledger,
            spool,
            device_inventory=device_inventory,
            **kwargs,
        )


__all__ = ["JobCoordinator"]
