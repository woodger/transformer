from __future__ import annotations

from app.service.adapters.inbound.flight.constants import (
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
)
from app.service.adapters.inbound.flight.output import OutputHandler
from app.service.adapters.inbound.flight.upload import UploadHandler
from app.service.adapters.outbound.postgres.input_uploads import (
    PostgresInputUploadStore,
)
from app.service.adapters.outbound.postgres.output_access import (
    PostgresOutputAccessStore,
)
from app.service.application.queries.outputs import OutputAccess
from app.service.application.services.input_upload import InputUploadLifecycle


def build_upload_handler(
    config,
    ledger,
    runtime_store,
    recovery_store,
    *,
    cuda_available,
    queue_notifier=None,
    input_notifier=None,
    metrics=None,
    logger=None,
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
    config,
    ledger,
    artifact_store,
    *,
    metrics=None,
    logger=None,
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
