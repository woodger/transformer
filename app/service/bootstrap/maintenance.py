from __future__ import annotations

from collections.abc import Callable

from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.application.services.maintenance import (
    MaintenanceService as MaintenanceApplicationService,
)


class MaintenanceService(MaintenanceApplicationService):
    """Compatibility composition facade for the maintenance use case."""

    def __init__(
        self,
        config,
        ledger,
        spool,
        recovery_store=None,
        *,
        interval_seconds: float = 60.0,
        queue_reconciler: Callable[[], None] | None = None,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
    ):
        super().__init__(
            config,
            ledger,
            spool,
            recovery_store,
            interval_seconds=interval_seconds,
            queue_reconciler=queue_reconciler,
            logger=logger or JsonLogger(),
            metrics=metrics or OperationalMetrics(),
        )


__all__ = ["MaintenanceService"]
