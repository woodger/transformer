import signal
import threading

from app.database import Database, load_database_config
from app.database.tokens import AccessTokenStore
from app.flight.auth import InMemoryAccessTokenCache
from app.flight.config import load_config
from app.flight.coordinator import JobCoordinator
from app.flight.ledger import Ledger
from app.flight.maintenance import MaintenanceService
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.output import OutputHandler
from app.flight.process import recover_process_groups
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool
from app.flight.token_cache import AccessTokenCache, AccessTokenCacheService
from app.flight.upload import UploadHandler
from app.flight.worker import WorkerPool


class FlightApplication:
    def __init__(
        self,
        config,
        spool,
        ledger,
        coordinator,
        server,
        maintenance,
        worker,
        metrics,
        logger,
        token_cache_service=None,
    ):
        self.config = config
        self.spool = spool
        self.ledger = ledger
        self.coordinator = coordinator
        self.server = server
        self.maintenance = maintenance
        self.worker = worker
        self.metrics = metrics
        self.logger = logger
        self.token_cache_service = token_cache_service
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = threading.Event()

    @classmethod
    def build(
        cls,
        config,
        *,
        database_config=None,
        models_dir=None,
        token_cache=None,
        bearer_tokens=None,
        logger=None,
    ):
        logger = logger or JsonLogger()
        metrics = OperationalMetrics()
        spool = Spool(config.runtime_dir, models_dir or config.models_dir).initialize()
        spool.acquire_lock()
        ledger = None
        token_cache_service = None
        worker = None
        coordinator = None
        server = None
        maintenance = None
        try:
            precleaned = spool.cleanup_temporary_files()
            database_config = database_config or load_database_config()
            ledger = Ledger(Database(database_config)).initialize()
            epoch_result = ledger.synchronize_runtime_epoch(spool.storage_epoch())
            process_recovery = recover_process_groups(
                ledger.list_recoverable_attempts(),
                grace_seconds=config.cancel_grace_seconds,
                logger=logger,
            )
            recovery = ledger.reconcile_interrupted_jobs()
            reconciliation = spool.reconcile(
                ledger.referenced_paths(),
                temporary_paths=recovery["temporary_paths"],
                known_job_ids={job["job_id"] for job in ledger.list_jobs()},
            )
            removed_models = spool.reconcile_model_directories(
                {model["model_ref"] for model in ledger.list_models()}
            )
            if token_cache is None and bearer_tokens is not None:
                token_cache = InMemoryAccessTokenCache(bearer_tokens)
            if token_cache is None:
                token_cache = AccessTokenCache()
                token_cache_service = AccessTokenCacheService(
                    database_config,
                    AccessTokenStore(ledger.database),
                    token_cache,
                    logger=logger,
                ).start()
            worker = WorkerPool(
                config,
                ledger,
                spool,
                metrics=metrics,
                logger=logger,
            )
            coordinator = JobCoordinator(
                config,
                ledger,
                spool,
                metrics=metrics,
                logger=logger,
                cancel_notifier=worker.notify_cancel,
                queue_notifier=worker.notify_queued,
            )
            upload = UploadHandler(
                config,
                ledger,
                spool,
                metrics=metrics,
                logger=logger,
            )
            output = OutputHandler(
                config,
                ledger,
                spool,
                metrics=metrics,
                logger=logger,
            )
            server = TransformerFlightServer(
                config,
                coordinator,
                token_cache,
                upload_handler=upload,
                output_handler=output,
                metrics=metrics,
                logger=logger,
            )
            maintenance = MaintenanceService(
                config,
                ledger,
                spool,
                interval_seconds=config.maintenance_interval_seconds,
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
                token_cache_service,
            )
            worker.start()
            maintenance.start()
            for job_id in recovery["interrupted_jobs"]:
                metrics.record_transition("RUNNING", "FAILED")
                logger.event(
                    "flight.job.transition",
                    jobId=job_id,
                    fromState="RUNNING",
                    toState="FAILED",
                    code="EXECUTION_INTERRUPTED",
                    recovery=True,
                )
            for job_id in recovery["cancelled_jobs"]:
                metrics.record_transition("CANCELLING", "CANCELLED")
                logger.event(
                    "flight.job.transition",
                    jobId=job_id,
                    fromState="CANCELLING",
                    toState="CANCELLED",
                    recovery=True,
                )
            usage = spool.disk_usage()
            logger.event(
                "flight.service.started",
                host=config.host,
                port=server.port,
                tls=config.tls_enabled,
                recoveredInterruptedJobs=len(recovery["interrupted_jobs"]),
                recoveredProcessGroups=sum(
                    result.outcome in ("terminated", "killed")
                    for result in process_recovery
                ),
                removedOrphans=len(reconciliation["removed"]),
                removedUnpublishedModels=len(removed_models),
                removedStartupTemporaries=len(precleaned),
                runtimeStorageReset=epoch_result["reset"],
                discardedRuntimeJobs=len(epoch_result["discarded_jobs"]),
                diskTotalBytes=usage.total,
                diskUsedBytes=usage.used,
                diskFreeBytes=usage.free,
                diskWatermarkBytes=config.disk_min_free_bytes,
                diskWatermarkExceeded=(
                    usage.free < config.disk_min_free_bytes
                ),
            )
        except BaseException:
            _cleanup_runtime(
                server=server,
                worker=worker,
                maintenance=maintenance,
                token_cache_service=token_cache_service,
                ledger=ledger,
                spool=spool,
                worker_timeout=0,
                maintenance_timeout=0,
            )
            raise
        return application

    def serve(self):
        previous = {}
        server_done = threading.Event()
        server_errors = []

        def run_server():
            try:
                self.server.serve()
            except BaseException as exc:
                server_errors.append(exc)
            finally:
                server_done.set()

        def handle_signal(signum, _frame):
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
                signal.signal(signum, handler)
            with self._shutdown_lock:
                shutdown_started = self._shutdown_started
            if not shutdown_started:
                self.shutdown()
            else:
                self._shutdown_complete.wait()
            server_thread.join()
        if server_errors:
            raise server_errors[0]

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_started:
                started_here = False
            else:
                self._shutdown_started = True
                started_here = True
        if not started_here:
            self._shutdown_complete.wait()
            return
        errors = []
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
                token_cache_service=self.token_cache_service,
                ledger=self.ledger,
                spool=self.spool,
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


def run_from_args(args):
    overrides = {
        "host": args.host,
        "port": args.port,
        "allow_plaintext": args.allow_plaintext,
        "tls_cert_file": args.tls_cert_file,
        "tls_key_file": args.tls_key_file,
        "tls_ca_file": args.tls_ca_file,
        "tls_require_client_cert": args.tls_require_client_cert,
    }
    config = load_config(overrides=overrides)
    FlightApplication.build(config).serve()


def _cleanup_runtime(
    *,
    server,
    worker,
    maintenance,
    token_cache_service,
    ledger,
    spool,
    worker_timeout,
    maintenance_timeout,
) -> list[BaseException]:
    """Stop partially or fully constructed runtime components in safe order."""
    errors: list[BaseException] = []
    operations = (
        None if server is None else server.shutdown,
        None if worker is None else lambda: worker.shutdown(worker_timeout),
        None if maintenance is None else lambda: maintenance.shutdown(maintenance_timeout),
        None if token_cache_service is None else lambda: token_cache_service.shutdown(maintenance_timeout),
        None if ledger is None else ledger.close,
        spool.release_lock,
    )
    for operation in operations:
        if operation is None:
            continue
        try:
            operation()
        except BaseException as exc:
            errors.append(exc)
    return errors
