from __future__ import annotations

from collections.abc import Callable

from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.application.ports.models import ModelDeletionRepository
from app.service.application.services.maintenance import (
    MaintenanceService as MaintenanceApplicationService,
)
from app.service.bootstrap.config import FlightServiceConfig


class MaintenanceService(MaintenanceApplicationService):
    """Фасад совместимой композиции для сценария обслуживания."""

    def __init__(
        self,
        config: FlightServiceConfig,
        ledger: Ledger,
        spool: Spool,
        recovery_store: RecoveryStore | None = None,
        *,
        model_deletions: ModelDeletionRepository | None = None,
        interval_seconds: float = 60.0,
        queue_reconciler: Callable[[], None] | None = None,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
    ) -> None:
        super().__init__(
            config,
            ledger,
            spool,
            recovery_store,
            model_deletions=model_deletions,
            interval_seconds=interval_seconds,
            queue_reconciler=queue_reconciler,
            logger=logger or JsonLogger(),
            metrics=metrics or OperationalMetrics(),
        )


__all__ = ["MaintenanceService"]
