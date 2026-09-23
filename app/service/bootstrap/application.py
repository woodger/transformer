from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import Protocol, cast

from app.contracts.json_types import JsonObject
from app.service.adapters.inbound.flight.server import TransformerFlightServer
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.artifacts.telemetry.projection import (
    TrainingMetricsProjection,
)
from app.service.adapters.outbound.cuda.inventory import (
    CudaDeviceInventory,
)
from app.service.adapters.outbound.opensearch.client import (
    OpenSearchMetricsClient,
)
from app.service.adapters.outbound.opensearch.config import (
    load_opensearch_metrics_config,
)
from app.service.adapters.outbound.opensearch.training_telemetry import (
    OpenSearchTrainingTelemetrySource,
    UnavailableTrainingTelemetrySource,
)
from app.service.adapters.outbound.postgres.config import (
    DatabaseConfig,
    load_database_config,
)
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.mapping import row_string
from app.service.adapters.outbound.postgres.published_models import (
    PublishedModelStore,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.adapters.outbound.postgres.telemetry import (
    PostgresMetricsOutbox,
    PostgresTrainingTelemetry,
)
from app.service.adapters.outbound.postgres.token_cache import AccessTokenCache
from app.service.adapters.outbound.postgres.tokens import AccessTokenStore
from app.service.adapters.outbound.worker.process import recover_process_groups
from app.service.application.ports.devices import DeviceLeaseManager
from app.service.application.telemetry.publisher import MetricsPublisher
from app.service.bootstrap.build_identity import load_build_identity
from app.service.bootstrap.config import FlightServiceConfig, load_config
from app.service.bootstrap.control_plane import build_job_coordinator
from app.service.bootstrap.data_plane import (
    build_output_handler,
    build_upload_handler,
)
from app.service.bootstrap.execution import WorkerPool
from app.service.bootstrap.maintenance import MaintenanceService


class FlightServiceArguments(Protocol):
    host: str | None
    port: int | None
    tls_cert_file: str | None
    tls_key_file: str | None
    tls_ca_file: str | None
    tls_require_client_cert: bool | None


class _CoordinatorRuntime(Protocol):
    def set_draining(self, value: bool = True) -> None: ...


class _FlightServerRuntime(Protocol):
    @property
    def port(self) -> int: ...

    def serve(self) -> None: ...

    def shutdown(self) -> None: ...


class _WorkerRuntime(Protocol):
    def stop_claiming(self) -> None: ...

    def shutdown(self, timeout: float | None = None) -> None: ...


class _MaintenanceRuntime(Protocol):
    def shutdown(self, timeout: float | None = None) -> None: ...


class _MetricsPublisherRuntime(Protocol):
    def shutdown(self, timeout: float | None = None) -> None: ...


class _LedgerRuntime(Protocol):
    def close(self) -> None: ...


class _LockRuntime(Protocol):
    def release_lock(self) -> None: ...


class FlightApplication:
    """Own service startup reconciliation, process runtime, and shutdown.

    A built instance holds the spool and recovery locks until shutdown has
    finished stopping every service-owned background component.
    """

    def __init__(
        self,
        config: FlightServiceConfig,
        spool: Spool,
        ledger: Ledger,
        coordinator: _CoordinatorRuntime,
        server: _FlightServerRuntime,
        maintenance: _MaintenanceRuntime,
        worker: _WorkerRuntime,
        metrics: OperationalMetrics,
        logger: JsonLogger,
        recovery_store: _LockRuntime | None = None,
        metrics_publisher: _MetricsPublisherRuntime | None = None,
    ) -> None:
        self.config = config
        self.spool = spool
        self.ledger = ledger
        self.coordinator = coordinator
        self.server = server
        self.maintenance = maintenance
        self.worker = worker
        self.metrics = metrics
        self.logger = logger
        self.recovery_store = recovery_store
        self.metrics_publisher = metrics_publisher
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = threading.Event()

    @classmethod
    def build(
        cls,
        config: FlightServiceConfig,
        *,
        database_config: DatabaseConfig | None = None,
        models_dir: str | os.PathLike[str] | None = None,
        logger: JsonLogger | None = None,
        device_inventory: DeviceLeaseManager | None = None,
    ) -> FlightApplication:
        logger = logger or JsonLogger()
        metrics = OperationalMetrics()
        model_path = config.models_dir if models_dir is None else models_dir
        model_path_text = (
            model_path
            if isinstance(model_path, str)
            else model_path.__fspath__()
        )
        effective_models_dir = os.path.abspath(model_path_text)
        telemetry_dir = (
            config.telemetry_dir
            if models_dir is None
            else os.path.join(
                os.path.dirname(effective_models_dir),
                "telemetry",
            )
        )
        spool = Spool(
            config.runtime_dir,
            effective_models_dir,
            telemetry_dir,
        ).initialize()
        recovery_dir = (
            config.recovery_dir
            if models_dir is None
            else os.path.join(
                os.path.dirname(effective_models_dir),
                "recovery",
            )
        )
        recovery_store = RecoveryStore(recovery_dir).initialize()
        ledger: Ledger | None = None
        worker: WorkerPool | None = None
        coordinator = None
        server: _FlightServerRuntime | None = None
        maintenance: MaintenanceService | None = None
        metrics_publisher: MetricsPublisher | None = None
        application: FlightApplication | None = None
        try:
            recovery_store.acquire_lock()
            spool.acquire_lock()
            database_config = database_config or load_database_config()
            ledger = Ledger(
                Database(database_config),
                logger=logger,
            ).initialize()
            published_models = PublishedModelStore(ledger.database)
            build_identity = load_build_identity()
            metrics_outbox = PostgresMetricsOutbox(ledger.database)
            training_telemetry = PostgresTrainingTelemetry(ledger.database)
            metrics_configuration_failed = False
            try:
                metrics_config = load_opensearch_metrics_config()
                metrics_client = (
                    None
                    if metrics_config is None
                    else OpenSearchMetricsClient(metrics_config)
                )
            except Exception as exc:
                metrics_configuration_failed = True
                metrics_config = None
                metrics_client = None
                metrics.add("metricsPublisherConfigurationErrors")
                logger.event(
                    "metrics.publisher.disabled",
                    reason="invalid-configuration",
                    errorType=type(exc).__name__,
                )
            if metrics_config is None:
                try:
                    backlog_entries, backlog_bytes, backlog_age = (
                        metrics_outbox.backlog()
                    )
                    metrics.set("metricsOutboxEntries", backlog_entries)
                    metrics.set("metricsOutboxBytes", backlog_bytes)
                    metrics.set("metricsOutboxOldestAgeSeconds", backlog_age)
                except Exception as exc:
                    backlog_entries = 0
                    metrics.add("metricsPublisherInternalErrors")
                    logger.event(
                        "metrics.publisher.disabled",
                        reason="outbox-unavailable",
                        errorType=type(exc).__name__,
                    )
                    metrics_configuration_failed = True
                if not metrics_configuration_failed:
                    logger.event(
                        "metrics.publisher.disabled",
                        pendingEntries=backlog_entries,
                    )
            else:
                if metrics_client is None:
                    raise AssertionError("configured metrics client is unavailable")
                metrics_publisher = MetricsPublisher(
                    metrics_outbox,
                    TrainingMetricsProjection(spool),
                    metrics_client,
                    spool,
                    deployment_id=metrics_config.deployment_id,
                    logger=logger,
                    metrics=metrics,
                )
            if metrics_client is None or metrics_config is None:
                training_telemetry_source = (
                    UnavailableTrainingTelemetrySource()
                )
            else:
                if metrics_publisher is None:
                    raise AssertionError(
                        "configured metrics publisher is unavailable"
                    )
                training_telemetry_source = OpenSearchTrainingTelemetrySource(
                    metrics_client,
                    metrics_outbox,
                    deployment_id=metrics_config.deployment_id,
                    delivery_expected=metrics_publisher.is_running,
                )
            # Завершаем процессы предыдущего сервиса до терминализации заданий
            # и удаления артефактов, которые они ещё могли записывать.
            process_recovery = recover_process_groups(
                ledger.list_recoverable_attempts(),
                grace_seconds=config.cancel_grace_seconds,
                logger=logger,
            )
            precleaned = spool.cleanup_temporary_files()
            recovery_precleaned = (
                recovery_store.cleanup_temporary_files()
            )
            epoch_result = ledger.synchronize_runtime_epoch(
                spool.storage_epoch()
            )
            startup_reconciliation = ledger.reconcile_startup_jobs()
            temporary_paths = _string_list(
                startup_reconciliation,
                "temporary_paths",
            )
            recovery_temporary_paths = _string_list(
                startup_reconciliation,
                "recovery_temporary_paths",
            )
            startup_failures = (
                (
                    "WAITING_INPUT",
                    _string_list(
                        startup_reconciliation,
                        "failed_waiting_input_jobs",
                    ),
                ),
                (
                    "QUEUED",
                    _string_list(
                        startup_reconciliation,
                        "failed_queued_jobs",
                    ),
                ),
                (
                    "RUNNING",
                    _string_list(
                        startup_reconciliation,
                        "failed_running_jobs",
                    ),
                ),
                (
                    "RETRYING",
                    _string_list(
                        startup_reconciliation,
                        "failed_retrying_jobs",
                    ),
                ),
            )
            cancelled_jobs = _string_list(
                startup_reconciliation,
                "cancelled_jobs",
            )
            reconciliation = spool.reconcile(
                ledger.referenced_paths(),
                temporary_paths=temporary_paths,
                known_job_ids={
                    row_string(job, "job_id")
                    for job in ledger.list_jobs()
                },
            )
            recovery_reconciliation = recovery_store.reconcile(
                ledger.recovery_referenced_paths(),
                known_job_ids=ledger.active_recovery_job_ids(),
                temporary_paths=recovery_temporary_paths,
            )
            removed_models = spool.reconcile_model_directories(
                published_models.retained_model_refs()
            )
            removed_telemetry_runs = spool.reconcile_telemetry_directories(
                metrics_outbox.retained_run_ids()
            )
            token_cache = AccessTokenCache(
                AccessTokenStore(ledger.database)
            )
            if device_inventory is None:
                device_inventory = CudaDeviceInventory(
                    logger=logger,
                    quarantine_path=os.path.join(
                        config.runtime_dir,
                        "cuda-quarantine.json",
                    ),
                ).initialize()
            worker = WorkerPool(
                config,
                ledger,
                spool,
                recovery_store,
                telemetry=training_telemetry,
                metrics=metrics,
                logger=logger,
                device_inventory=device_inventory,
                build_identity=build_identity,
            )
            coordinator = build_job_coordinator(
                config,
                ledger,
                spool,
                recovery_store,
                metrics=metrics,
                logger=logger,
                training_telemetry_source=training_telemetry_source,
                cancel_notifier=worker.notify_cancel,
                queue_notifier=worker.notify_queued,
                device_inventory=device_inventory,
            )
            upload = build_upload_handler(
                config,
                ledger,
                spool,
                recovery_store,
                cuda_available=lambda: (
                    device_inventory.snapshot().cuda_capacity > 0
                ),
                queue_notifier=worker.notify_queued,
                input_notifier=worker.notify_input,
                metrics=metrics,
                logger=logger,
            )
            output = build_output_handler(
                config,
                ledger,
                spool,
                metrics=metrics,
                logger=logger,
            )
            server = cast(
                _FlightServerRuntime,
                TransformerFlightServer(
                    config,
                    coordinator,
                    token_cache,
                    upload_handler=upload,
                    output_handler=output,
                    metrics=metrics,
                    logger=logger,
                ),
            )
            maintenance = MaintenanceService(
                config,
                ledger,
                spool,
                recovery_store,
                model_deletions=published_models,
                interval_seconds=config.maintenance_interval_seconds,
                queue_reconciler=worker.notify_queued,
                logger=logger,
                metrics=metrics,
            )
            application = cls(
                config,
                spool,
                ledger,
                coordinator,
                server,
                maintenance,
                worker,
                metrics,
                logger,
                recovery_store=recovery_store,
                metrics_publisher=metrics_publisher,
            )
            if metrics_publisher is not None:
                try:
                    metrics_publisher.start()
                except Exception as exc:
                    metrics.add("metricsPublisherInternalErrors")
                    logger.event(
                        "metrics.publisher.disabled",
                        reason="startup-failed",
                        errorType=type(exc).__name__,
                    )
                    metrics_publisher = None
                    application.metrics_publisher = None
            worker.start()
            maintenance.start()
            for from_state, job_ids in startup_failures:
                for job_id in job_ids:
                    metrics.record_transition(from_state, "FAILED")
                    logger.event(
                        "flight.job.transition",
                        jobId=job_id,
                        fromState=from_state,
                        toState="FAILED",
                        code="EXECUTION_INTERRUPTED",
                        startup=True,
                    )
            for job_id in cancelled_jobs:
                metrics.record_transition("CANCELLING", "CANCELLED")
                logger.event(
                    "flight.job.transition",
                    jobId=job_id,
                    fromState="CANCELLING",
                    toState="CANCELLED",
                    startup=True,
                )
            usage = spool.disk_usage()
            recovery_usage = recovery_store.disk_usage()
            logger.event(
                "flight.service.started",
                host=config.host,
                port=server.port,
                tls=config.tls_enabled,
                failedStartupJobs=sum(
                    len(job_ids) for _, job_ids in startup_failures
                ),
                cancelledStartupJobs=len(cancelled_jobs),
                recoveredProcessGroups=sum(
                    result.outcome in ("terminated", "killed")
                    for result in process_recovery
                ),
                removedOrphans=len(_string_list(reconciliation, "removed")),
                removedUnpublishedModels=len(removed_models),
                removedTelemetryRuns=len(removed_telemetry_runs),
                removedStartupTemporaries=len(precleaned),
                removedRecoveryTemporaries=len(
                    recovery_precleaned
                ),
                removedRecoveryOrphans=len(
                    recovery_reconciliation
                ),
                runtimeStorageReset=_boolean(epoch_result, "reset"),
                discardedRuntimeJobs=len(
                    _string_list(epoch_result, "discarded_jobs")
                ),
                diskTotalBytes=usage.total,
                diskUsedBytes=usage.used,
                diskFreeBytes=usage.free,
                recoveryDiskTotalBytes=recovery_usage.total,
                recoveryDiskUsedBytes=recovery_usage.used,
                recoveryDiskFreeBytes=recovery_usage.free,
                cudaDevices=(
                    device_inventory.snapshot().device_count
                ),
                cudaCapacity=(
                    device_inventory.snapshot().cuda_capacity
                ),
            )
        except BaseException:
            _cleanup_runtime(
                server=server,
                worker=worker,
                maintenance=maintenance,
                metrics_publisher=metrics_publisher,
                ledger=ledger,
                spool=spool,
                recovery_store=recovery_store,
                worker_timeout=0,
                maintenance_timeout=0,
            )
            raise
        return application

    def serve(self) -> None:
        previous: dict[signal.Signals, object] = {}
        server_done = threading.Event()
        server_errors: list[BaseException] = []

        def run_server() -> None:
            try:
                self.server.serve()
            except BaseException as exc:
                server_errors.append(exc)
            finally:
                server_done.set()

        def handle_signal(signum: int, _frame: FrameType | None) -> None:
            self.logger.event("flight.service.signal", signal=signum)
            # Keep shutdown outside the signal handler and outside Flight RPC
            # threads.  The non-daemon thread guarantees process exit cannot
            # skip durable worker, ledger, and state-lock cleanup.
            threading.Thread(
                target=self.shutdown,
                name="transformer-flight-shutdown",
                daemon=False,
            ).start()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
        server_thread = threading.Thread(
            target=run_server,
            name="transformer-flight-server",
            daemon=False,
        )
        try:
            # FlightServerBase.serve() is a blocking C-extension call.  Run it
            # off the Python main thread so SIGINT/SIGTERM handlers are
            # dispatched promptly even while no RPC is active.
            server_thread.start()
            self.logger.event(
                "flight.service.serving",
                host=self.config.host,
                port=self.server.port,
                tls=self.config.tls_enabled,
            )
            server_done.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(
                    signum,
                    cast(Callable[[int, FrameType | None], None], handler),
                )
            with self._shutdown_lock:
                shutdown_started = self._shutdown_started
            if not shutdown_started:
                self.shutdown()
            else:
                self._shutdown_complete.wait()
            server_thread.join()
        if server_errors:
            raise server_errors[0]

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                started_here = False
            else:
                self._shutdown_started = True
                started_here = True
        if not started_here:
            self._shutdown_complete.wait()
            return
        errors: list[BaseException] = []
        try:
            # Close the queue claim boundary before RPC shutdown.  A start
            # already racing with this point may still commit QUEUED, but it
            # cannot become RUNNING and is therefore safe to recover on the
            # next service start.
            try:
                stop_claiming = getattr(self.worker, "stop_claiming", None)
                if stop_claiming is not None:
                    stop_claiming()
            except BaseException as exc:
                errors.append(exc)
            try:
                self.coordinator.set_draining(True)
            except BaseException as exc:
                errors.append(exc)
            try:
                self.logger.event("flight.service.draining")
            except BaseException as exc:
                errors.append(exc)
            errors.extend(_cleanup_runtime(
                server=self.server,
                worker=self.worker,
                maintenance=self.maintenance,
                metrics_publisher=self.metrics_publisher,
                ledger=self.ledger,
                spool=self.spool,
                recovery_store=self.recovery_store,
                worker_timeout=self.config.shutdown_drain_seconds,
                maintenance_timeout=self.config.shutdown_drain_seconds,
            ))
        finally:
            try:
                self.logger.event("flight.service.stopped")
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._shutdown_complete.set()
        if errors:
            raise errors[0]


def run_from_args(args: FlightServiceArguments) -> None:
    overrides = {
        "host": args.host,
        "port": args.port,
        "tls_cert_file": args.tls_cert_file,
        "tls_key_file": args.tls_key_file,
        "tls_ca_file": args.tls_ca_file,
        "tls_require_client_cert": args.tls_require_client_cert,
    }
    config = load_config(overrides=overrides)
    FlightApplication.build(config).serve()


def _cleanup_runtime(
    *,
    server: _FlightServerRuntime | None,
    worker: _WorkerRuntime | None,
    maintenance: _MaintenanceRuntime | None,
    metrics_publisher: _MetricsPublisherRuntime | None,
    ledger: _LedgerRuntime | None,
    spool: _LockRuntime,
    recovery_store: _LockRuntime | None = None,
    worker_timeout: float,
    maintenance_timeout: float,
) -> list[BaseException]:
    """Stop partially or fully constructed runtime components in safe order."""
    errors: list[BaseException] = []
    operations: tuple[Callable[[], None] | None, ...] = (
        None if server is None else server.shutdown,
        None if worker is None else lambda: worker.shutdown(worker_timeout),
        (
            None
            if metrics_publisher is None
            else lambda: metrics_publisher.shutdown(maintenance_timeout)
        ),
        None if maintenance is None else lambda: maintenance.shutdown(maintenance_timeout),
        None if ledger is None else ledger.close,
        spool.release_lock,
        (
            None
            if recovery_store is None
            else recovery_store.release_lock
        ),
    )
    for operation in operations:
        if operation is None:
            continue
        try:
            operation()
        except BaseException as exc:
            errors.append(exc)
    return errors


def _string_list(document: JsonObject, key: str) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"service startup field {key} must be a list")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(
            f"service startup field {key} must contain strings"
        )
    return tuple(cast(str, item) for item in items)


def _boolean(document: JsonObject, key: str) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"service startup field {key} must be a boolean")
    return value
