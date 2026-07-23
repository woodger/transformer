import pyarrow
import torch

from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CONTRACT_VERSION,
    CREATE_ACTION,
    ErrorCode,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    PREDICTION_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
)
from app.flight.contract import encode_document, response_document
from app.flight.errors import ServiceError
from app.flight.job_actions import CreateStatusActions, LifecycleActions
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.spool import Spool
from app.runtime.version import __version__


class JobCoordinator:
    def __init__(
        self,
        config: FlightServiceConfig,
        ledger,
        spool: Spool,
        *,
        cuda_available=None,
        metrics: OperationalMetrics | None = None,
        logger: JsonLogger | None = None,
        cancel_notifier=None,
        queue_notifier=None,
    ):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self._cuda_available = cuda_available or torch.cuda.is_available
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()
        self.cancel_notifier = cancel_notifier
        self.queue_notifier = queue_notifier
        self.draining = False
        self._create_status_actions = CreateStatusActions(
            config,
            ledger,
            spool,
            cuda_available=lambda: self._cuda_available(),
            is_draining=lambda: self.draining,
            limits=self._limits,
            metrics=self.metrics,
            logger=self.logger,
        )
        self._lifecycle_actions = LifecycleActions(
            ledger,
            cuda_available=lambda: self._cuda_available(),
            is_draining=lambda: self.draining,
            queue_notifier=self._notify_queued,
            cancel_notifier=self._notify_cancel,
            metrics=self.metrics,
            logger=self.logger,
        )

    def dispatch(self, action: str, owner: str, request: dict, document: dict) -> bytes:
        if action == CAPABILITIES_ACTION:
            result = self.capabilities(request["request_id"])
        elif action == HEALTH_ACTION:
            result = self.health(request["request_id"])
        elif action == CREATE_ACTION:
            result = self.create(owner, request, document)
        elif action == STATUS_ACTION:
            result = self.status(owner, request["job_id"], request["request_id"])
        elif action == SEAL_ACTION:
            result = self.seal(owner, request, document)
        elif action == START_ACTION:
            result = self.start(owner, request, document)
        elif action == CANCEL_ACTION:
            result = self.cancel(owner, request, document)
        else:  # Contract validation normally rejects this first.
            raise ServiceError(ErrorCode.INVALID_ARGUMENT, f"unsupported action: {action}")
        return encode_document(result)

    def capabilities(self, request_id: str) -> dict:
        cuda_available = bool(self._cuda_available())
        self.metrics.set("cudaAvailable", cuda_available)
        return response_document(
            request_id,
            protocolVersions=[CONTRACT_VERSION],
            service={
                "name": "transformer-flight",
                "version": __version__,
                "pyarrowVersion": pyarrow.__version__,
                "torchVersion": torch.__version__,
            },
            schemaIds={
                "fitInput": FIT_SCHEMA_ID,
                "predictInput": PREDICT_SCHEMA_ID,
                "predictionOutput": PREDICTION_SCHEMA_ID,
            },
            limits=self._limits(),
            devices={
                "cpu": {"available": True},
                "cuda": {
                    "available": cuda_available,
                    "deviceCount": torch.cuda.device_count() if cuda_available else 0,
                    "runtimeVersion": torch.version.cuda,
                },
            },
            queue={
                "cpuCapacity": self.config.cpu_capacity,
                "cudaCapacity": self.config.cuda_capacity,
                "singleInstance": True,
            },
            supportedOperations=["fit", "predict"],
            features={"doExchange": False, "pollFlightInfo": False},
        )

    def health(self, request_id: str) -> dict:
        usage = self.spool.disk_usage()
        try:
            ledger_ready = bool(self.ledger.healthcheck())
        except Exception as exc:
            ledger_ready = False
            self.logger.event(
                "flight.ledger.health_failed",
                errorType=type(exc).__name__,
            )
        ready = (
            not self.draining
            and ledger_ready
            and usage.free >= self.config.disk_min_free_bytes
        )
        cuda_available = bool(self._cuda_available())
        self.metrics.set("cudaAvailable", cuda_available)
        self.metrics.set("ready", ready)
        self.metrics.set("diskTotalBytes", usage.total)
        self.metrics.set("diskUsedBytes", usage.used)
        self.metrics.set("diskFreeBytes", usage.free)
        self.metrics.set("diskWatermarkBytes", self.config.disk_min_free_bytes)
        self.metrics.set(
            "diskWatermarkExceeded",
            usage.free < self.config.disk_min_free_bytes,
        )
        return response_document(
            request_id,
            live=True,
            ready=ready,
            draining=self.draining,
            ledger={"available": ledger_ready},
            cuda={"available": cuda_available},
            disk={
                "freeBytes": usage.free,
                "watermarkBytes": self.config.disk_min_free_bytes,
                "watermarkExceeded": usage.free < self.config.disk_min_free_bytes,
            },
            metrics=self.metrics.snapshot(),
        )

    def create(self, owner: str, request: dict, document: dict) -> dict:
        return self._create_status_actions.create(owner, request, document)

    def seal(self, owner: str, request: dict, document: dict) -> dict:
        return self._lifecycle_actions.seal(owner, request, document)

    def start(self, owner: str, request: dict, document: dict) -> dict:
        return self._lifecycle_actions.start(owner, request, document)

    def cancel(self, owner: str, request: dict, document: dict) -> dict:
        return self._lifecycle_actions.cancel(owner, request, document)

    def status(self, owner: str, job_id: str, request_id: str) -> dict:
        return self._create_status_actions.status(
            owner,
            job_id,
            request_id,
        )

    def set_draining(self, value: bool = True) -> None:
        self.draining = bool(value)

    def _notify_queued(self, job_id: str) -> None:
        notifier = self.queue_notifier
        if notifier is not None:
            notifier(job_id)

    def _notify_cancel(self, job_id: str) -> None:
        notifier = self.cancel_notifier
        if notifier is not None:
            notifier(job_id)

    def _limits(self) -> dict:
        return {
            "maxMessageBytes": self.config.max_message_bytes,
            "targetBatchBytes": self.config.target_batch_bytes,
            "maxBatchBytes": self.config.max_batch_bytes,
            "maxPayloadBytes": self.config.max_payload_bytes,
            "maxRowsPerPayload": self.config.max_rows_per_payload,
            "maxPayloadsPerJob": self.config.max_payloads_per_job,
            "maxJobBytes": self.config.max_job_bytes,
            "maxActiveJobsPerSubject": self.config.max_active_jobs_per_subject,
            "transportMessageLimitEnforced": False,
        }
