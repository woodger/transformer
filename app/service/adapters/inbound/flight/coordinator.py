import hashlib
import os

import pyarrow

from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CONTRACT_PATH_VERSION,
    CONTRACT_VERSION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MAX_PAGE_ITEMS,
    MODEL_DESCRIBE_ACTION,
    OUTPUTS_LIST_ACTION,
    PREDICT_SCHEMA_ID,
    PREDICTION_SCHEMA_ID,
    STATUS_ACTION,
    ErrorCode,
)
from app.service.adapters.inbound.flight.contract import (
    canonical_request_hash,
    data_contract_to_api,
    encode_document,
    model_config_to_api,
    response_document,
)
from app.service.adapters.inbound.flight.errors import ServiceError
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.application.commands.jobs import (
    AcquireJobAction,
    CancelJobAction,
    CreateJobAction,
    InputCloseAction,
    JobActionContract,
)
from app.service.application.ports.devices import WorkerCapabilities
from app.service.application.queries.status import (
    DescribeModel,
    GetJobStatus,
    ListJobInputs,
    ListJobOutputs,
)
from app.version import __version__


class JobCoordinator:
    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        device_inventory: WorkerCapabilities,
        recovery_store=None,
        metrics: OperationalMetrics | None = None,
        logger: JsonLogger | None = None,
        cancel_notifier=None,
        queue_notifier=None,
    ):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.recovery_store = recovery_store or spool
        self.device_inventory = device_inventory
        self._cuda_available = lambda: (
            self.device_inventory.snapshot().cuda_capacity > 0
        )
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()
        self.cancel_notifier = cancel_notifier
        self.queue_notifier = queue_notifier
        self.draining = False
        contract = JobActionContract(
            create_action=CREATE_ACTION,
            acquire_action=ACQUIRE_ACTION,
            input_close_action=INPUT_CLOSE_ACTION,
            cancel_action=CANCEL_ACTION,
            path_version=CONTRACT_PATH_VERSION,
            fit_schema_id=FIT_SCHEMA_ID,
            predict_schema_id=PREDICT_SCHEMA_ID,
        )
        common = {
            "action_contract": contract,
            "request_hasher": canonical_request_hash,
            "response_factory": response_document,
        }
        self._create_action = CreateJobAction(
            config,
            ledger,
            cuda_available=lambda: self._cuda_available(),
            is_draining=lambda: self.draining,
            limits=self._limits,
            data_contract_factory=data_contract_to_api,
            model_validator=self._validate_model_artifact,
            metrics=self.metrics,
            logger=self.logger,
            **common,
        )
        self._acquire_action = AcquireJobAction(
            config,
            ledger,
            cleanup_candidate=self._cleanup_candidate,
            logger=self.logger,
            **common,
        )
        self._close_action = InputCloseAction(
            ledger,
            cuda_available=lambda: self._cuda_available(),
            queue_notifier=self._notify_queued,
            metrics=self.metrics,
            logger=self.logger,
            **common,
        )
        self._cancel_action = CancelJobAction(
            ledger,
            cancel_notifier=self._notify_cancel,
            cleanup_candidate=self._cleanup_candidate,
            metrics=self.metrics,
            logger=self.logger,
            **common,
        )
        self._status_query = GetJobStatus(
            ledger,
            response_factory=response_document,
            data_contract_factory=data_contract_to_api,
        )
        self._inputs_query = ListJobInputs(
            ledger,
            response_factory=response_document,
        )
        self._outputs_query = ListJobOutputs(
            ledger,
            path_version=CONTRACT_PATH_VERSION,
            response_factory=response_document,
        )
        self._model_query = DescribeModel(
            ledger,
            response_factory=response_document,
            data_contract_factory=data_contract_to_api,
            model_config_factory=model_config_to_api,
            model_artifact_validator=self._validate_model_artifact,
        )

    def dispatch(self, action: str, owner: str, request: dict, document: dict) -> bytes:
        if action == CAPABILITIES_ACTION:
            result = self.capabilities(request["request_id"])
        elif action == HEALTH_ACTION:
            result = self.health(request["request_id"])
        elif action == CREATE_ACTION:
            result = self._create_action.create(owner, request, document)
        elif action == ACQUIRE_ACTION:
            result = self._acquire_action.acquire(owner, request, document)
        elif action == STATUS_ACTION:
            result = self._status_query.execute(
                owner,
                request["job_id"],
                request["request_id"],
            )
        elif action == INPUTS_LIST_ACTION:
            result = self._inputs_query.execute(owner, request)
        elif action == INPUT_CLOSE_ACTION:
            result = self._close_action.close(owner, request, document)
        elif action == OUTPUTS_LIST_ACTION:
            result = self._outputs_query.execute(owner, request)
        elif action == CANCEL_ACTION:
            result = self._cancel_action.cancel(owner, request, document)
        elif action == MODEL_DESCRIBE_ACTION:
            result = self._model_query.execute(owner, request)
        else:
            raise ServiceError(
                ErrorCode.INVALID_ARGUMENT,
                f"unsupported action: {action}",
            )
        return encode_document(result)

    def capabilities(self, request_id: str) -> dict:
        inventory = self.device_inventory.snapshot()
        cuda_available = inventory.cuda_capacity > 0
        self.metrics.set("cudaAvailable", cuda_available)
        return response_document(
            request_id,
            protocolVersions=[CONTRACT_VERSION],
            service={
                "name": "transformer-flight",
                "version": __version__,
                "pyarrowVersion": pyarrow.__version__,
                "torchVersion": inventory.torch_version,
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
                    "deviceCount": inventory.device_count,
                    "quarantinedCount": inventory.quarantined_count,
                    "runtimeVersion": inventory.runtime_version,
                },
            },
            queue={
                "cpuCapacity": self.config.cpu_capacity,
                "cudaCapacity": inventory.cuda_capacity,
                "singleInstance": True,
            },
            supportedOperations=["fit", "predict"],
            features={
                "doExchange": False,
                "pollFlightInfo": False,
                "durableStreamingInput": True,
                "clientGeneratedJobId": True,
                "crossSystemFencing": True,
                "revisionPagination": True,
                "resumableFit": True,
                "recoveryBoundary": "globalEpoch",
                "deviceAwareCuda": True,
            },
        )

    def health(self, request_id: str) -> dict:
        runtime_usage = self.spool.disk_usage()
        recovery_usage = self.recovery_store.disk_usage()
        try:
            ledger_ready = bool(self.ledger.healthcheck())
        except Exception as exc:
            ledger_ready = False
            self.logger.event(
                "flight.ledger.health_failed",
                errorType=type(exc).__name__,
            )
        ready = not self.draining and ledger_ready
        inventory = self.device_inventory.snapshot()
        cuda_available = inventory.cuda_capacity > 0
        self.metrics.set("cudaAvailable", cuda_available)
        self.metrics.set("ready", ready)
        self.metrics.set("diskTotalBytes", runtime_usage.total)
        self.metrics.set("diskUsedBytes", runtime_usage.used)
        self.metrics.set("diskFreeBytes", runtime_usage.free)
        self.metrics.set("recoveryDiskTotalBytes", recovery_usage.total)
        self.metrics.set("recoveryDiskUsedBytes", recovery_usage.used)
        self.metrics.set("recoveryDiskFreeBytes", recovery_usage.free)
        return response_document(
            request_id,
            live=True,
            ready=ready,
            draining=self.draining,
            ledger={"available": ledger_ready},
            cuda={
                "available": cuda_available,
                "deviceCount": inventory.device_count,
                "quarantinedCount": inventory.quarantined_count,
            },
            storage={
                "runtime": _storage_health(runtime_usage),
                "recovery": _storage_health(recovery_usage),
            },
            metrics=self.metrics.snapshot(),
        )

    def set_draining(self, value: bool = True) -> None:
        self.draining = bool(value)

    def _notify_queued(self, job_id: str) -> None:
        if self.queue_notifier is not None:
            self.queue_notifier(job_id)

    def _notify_cancel(self, job_id: str) -> None:
        if self.cancel_notifier is not None:
            self.cancel_notifier(job_id)

    def _cleanup_candidate(self, storage_class: str, relative_path: str) -> None:
        store = self.recovery_store if storage_class == "recovery" else self.spool
        try:
            store.remove(store.absolute_path(relative_path))
        except (FileNotFoundError, OSError):
            self.logger.event(
                "flight.input.candidate_cleanup_failed",
                storageClass=storage_class,
                path=os.path.basename(relative_path),
            )

    def _validate_model_artifact(self, model) -> None:
        try:
            path = self.spool.model_absolute_path(model.checkpoint_path)
            expected = self.spool.model_checkpoint_path(model.model_ref)
        except ValueError as exc:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model checkpoint identity is invalid",
            ) from exc
        if path != expected:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model checkpoint identity is invalid",
            )
        try:
            byte_count = os.path.getsize(path)
            checkpoint_sha256 = _sha256_file(path)
        except OSError as exc:
            raise ServiceError(
                ErrorCode.MODEL_UNAVAILABLE,
                "model checkpoint is unavailable",
            ) from exc
        if byte_count != model.byte_count or checkpoint_sha256 != model.sha256:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model checkpoint integrity validation failed",
            )

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
            "maxPageItems": MAX_PAGE_ITEMS,
            "inputIdleTimeoutSeconds": self.config.input_idle_timeout_seconds,
            "transportMessageLimitEnforced": False,
        }


def _storage_health(usage) -> dict:
    return {"freeBytes": usage.free}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
