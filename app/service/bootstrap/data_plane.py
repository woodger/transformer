from __future__ import annotations

from collections.abc import Callable

from app.service.adapters.inbound.flight.constants import (
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
)
from app.service.adapters.inbound.flight.output import OutputHandler
from app.service.adapters.inbound.flight.upload import UploadHandler
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.postgres.input_uploads import (
    PostgresInputUploadStore,
)
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.output_access import (
    PostgresOutputAccessStore,
)
from app.service.application.queries.outputs import OutputAccess
from app.service.application.services.input_upload import InputUploadLifecycle
from app.service.bootstrap.config import FlightServiceConfig


def build_upload_handler(
    config: FlightServiceConfig,
    ledger: Ledger,
    runtime_store: Spool,
    recovery_store: RecoveryStore | None,
    *,
    cuda_available: Callable[[], bool],
    queue_notifier: Callable[[str], None] | None = None,
    input_notifier: Callable[[str], None] | None = None,
    metrics: OperationalMetrics | None = None,
    logger: JsonLogger | None = None,
) -> UploadHandler:
    lifecycle = InputUploadLifecycle(
        PostgresInputUploadStore(
            ledger,
            schema_ids={
                "fit": FIT_SCHEMA_ID,
                "predict": PREDICT_SCHEMA_ID,
            },
        ),
        max_payloads=config.max_payloads_per_job,
        max_job_bytes=config.max_job_bytes,
        recovery_enabled=recovery_store is not None,
        cuda_available=cuda_available,
    )
    return UploadHandler(
        config,
        lifecycle,
        {
            "runtime": runtime_store,
            "recovery": recovery_store or runtime_store,
        },
        queue_notifier=queue_notifier,
        input_notifier=input_notifier,
        metrics=metrics,
        logger=logger,
    )


def build_output_handler(
    config: FlightServiceConfig,
    ledger: Ledger,
    artifact_store: Spool,
    *,
    metrics: OperationalMetrics | None = None,
    logger: JsonLogger | None = None,
) -> OutputHandler:
    access = OutputAccess(
        PostgresOutputAccessStore(ledger),
        ticket_ttl_seconds=config.ticket_ttl_seconds,
    )
    return OutputHandler(
        access,
        artifact_store,
        metrics=metrics,
        logger=logger,
    )


__all__ = ["build_output_handler", "build_upload_handler"]
