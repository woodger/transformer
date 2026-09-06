from __future__ import annotations

from collections.abc import Callable

from app.contracts.flight.v12 import job_config_sha256
from app.contracts.model_catalog.v1 import (
    CURSOR_TTL_SECONDS,
    MAX_CHECKPOINT_VERIFICATION_BYTES,
)
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CREATE_ACTION,
    INPUT_CLOSE_ACTION,
    MAX_PAGE_ITEMS,
)
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.adapters.inbound.flight.model_catalog import (
    CatalogModelMetadataVerifier,
)
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.artifacts.job_artifacts import (
    CandidateArtifactCleaner,
    CatalogModelArtifactVerifier,
    ModelArtifactVerifier,
)
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.postgres.job_lifecycle import (
    JobActionNames,
    PostgresJobLifecycle,
)
from app.service.adapters.outbound.postgres.job_queries import (
    PostgresJobQueryStore,
)
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.model_catalog import (
    PostgresModelCatalogStore,
)
from app.service.application.commands.jobs import (
    AcquireJobAction,
    CancelJobAction,
    CreateJobAction,
    InputCloseAction,
)
from app.service.application.messages.jobs import ServiceLimits
from app.service.application.ports.devices import WorkerCapabilities
from app.service.application.queries.model_catalog import (
    GetCatalogModel,
    ListCatalogModels,
)
from app.service.application.queries.service import (
    ServiceAvailability,
    ServiceStatusQuery,
)
from app.service.application.queries.status import (
    GetJobStatus,
    ListJobInputs,
    ListJobOutputs,
)
from app.service.bootstrap.config import FlightServiceConfig


def build_job_coordinator(
    config: FlightServiceConfig,
    ledger: Ledger,
    spool: Spool,
    recovery_store: RecoveryStore,
    *,
    device_inventory: WorkerCapabilities,
    metrics: OperationalMetrics,
    logger: JsonLogger,
    cancel_notifier: Callable[[str], None] | None = None,
    queue_notifier: Callable[[str], None] | None = None,
) -> JobCoordinator:
    limits = _service_limits(config)
    availability = ServiceAvailability()
    lifecycle = PostgresJobLifecycle(
        ledger,
        JobActionNames(
            create=CREATE_ACTION,
            acquire=ACQUIRE_ACTION,
            input_close=INPUT_CLOSE_ACTION,
            cancel=CANCEL_ACTION,
        ),
    )
    queries = PostgresJobQueryStore(ledger)
    model_verifier = ModelArtifactVerifier(spool)
    model_catalog_store = PostgresModelCatalogStore(ledger.database)
    artifact_cleaner = CandidateArtifactCleaner(
        spool,
        recovery_store,
        logger=logger,
    )

    def cuda_available() -> bool:
        return device_inventory.snapshot().cuda_capacity > 0

    create_job = CreateJobAction(
        lifecycle,
        max_active_jobs=config.max_active_jobs_per_subject,
        limits=limits,
        cuda_available=cuda_available,
        is_draining=lambda: availability.draining,
        job_config_digest=job_config_sha256,
        model_verifier=model_verifier,
        metrics=metrics,
        logger=logger,
    )
    acquire_job = AcquireJobAction(
        lifecycle,
        acquire_grace_seconds=config.acquire_idle_grace_seconds,
        artifact_cleaner=artifact_cleaner,
        logger=logger,
    )
    close_input = InputCloseAction(
        lifecycle,
        cuda_available=cuda_available,
        queue_notifier=queue_notifier or (lambda _job_id: None),
        metrics=metrics,
        logger=logger,
    )
    cancel_job = CancelJobAction(
        lifecycle,
        cancel_notifier=cancel_notifier or (lambda _job_id: None),
        artifact_cleaner=artifact_cleaner,
        metrics=metrics,
        logger=logger,
    )
    service_status = ServiceStatusQuery(
        health_repository=ledger,
        runtime_storage=spool,
        recovery_storage=recovery_store,
        device_inventory=device_inventory,
        availability=availability,
        cpu_capacity=config.cpu_capacity,
        limits=limits,
        metrics=metrics,
        logger=logger,
    )
    return JobCoordinator(
        create_job=create_job,
        acquire_job=acquire_job,
        close_input=close_input,
        cancel_job=cancel_job,
        get_status=GetJobStatus(queries),
        list_inputs=ListJobInputs(queries),
        list_outputs=ListJobOutputs(queries),
        list_catalog_models=ListCatalogModels(
            model_catalog_store,
            cursor_ttl_seconds=CURSOR_TTL_SECONDS,
        ),
        get_catalog_model=GetCatalogModel(
            model_catalog_store,
            metadata_verifier=CatalogModelMetadataVerifier(),
            artifact_verifier=CatalogModelArtifactVerifier(
                spool,
                max_verification_bytes=MAX_CHECKPOINT_VERIFICATION_BYTES,
            ),
        ),
        service_status=service_status,
        availability=availability,
    )


def _service_limits(config: FlightServiceConfig) -> ServiceLimits:
    return ServiceLimits(
        max_message_bytes=config.max_message_bytes,
        target_batch_bytes=config.target_batch_bytes,
        max_batch_bytes=config.max_batch_bytes,
        max_payload_bytes=config.max_payload_bytes,
        max_rows_per_payload=config.max_rows_per_payload,
        max_payloads_per_job=config.max_payloads_per_job,
        max_job_bytes=config.max_job_bytes,
        max_active_jobs_per_subject=config.max_active_jobs_per_subject,
        max_page_items=MAX_PAGE_ITEMS,
        input_idle_timeout_seconds=config.input_idle_timeout_seconds,
    )


__all__ = ["build_job_coordinator"]
