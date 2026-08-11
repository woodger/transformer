from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.service.domain.errors import ServiceError, conflict, not_found
from app.service.domain.job import ErrorCode, ExecutionState
from app.service.domain.policies import resolve_device


@dataclass(frozen=True, slots=True)
class JobActionContract:
    create_action: str
    acquire_action: str
    input_close_action: str
    cancel_action: str
    path_version: str
    fit_schema_id: str
    predict_schema_id: str


class CreateJobAction:
    """Create a client-identified durable job in one transaction."""

    def __init__(
        self,
        config,
        ledger,
        *,
        cuda_available: Callable[[], bool],
        is_draining: Callable[[], bool],
        limits: Callable[[], dict],
        action_contract: JobActionContract,
        request_hasher: Callable[[dict], str],
        response_factory: Callable[..., dict],
        data_contract_factory: Callable[[dict], dict],
        model_validator: Callable[[object], None],
        metrics,
        logger,
    ):
        self.config = config
        self.ledger = ledger
        self._cuda_available = cuda_available
        self._is_draining = is_draining
        self._limits = limits
        self.contract = action_contract
        self._request_hash = request_hasher
        self._response = response_factory
        self._data_contract = data_contract_factory
        self._model_validator = model_validator
        self.metrics = metrics
        self.logger = logger

    def create(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = self._request_hash(document)
        replay = _replay(
            self.ledger,
            owner,
            self.contract.create_action,
            request["idempotency_key"],
            request_hash,
        )
        if replay is not None:
            return replay
        identity = self.ledger.get_job_identity(request["job_id"])
        if identity is not None:
            if (
                identity["owner_subject"] != owner
                or identity["create_hash"] != request_hash
            ):
                raise conflict(
                    "jobId has already been used for a different request"
                )
            return identity["create_result"]
        if self._is_draining():
            raise ServiceError(ErrorCode.UNAVAILABLE, "service is draining")
        if request["device"] == "cuda" and not self._cuda_available():
            self.metrics.add("cudaUnavailableRequests")
            raise ServiceError(
                ErrorCode.DEVICE_UNAVAILABLE,
                "explicit CUDA device is not available",
            )

        created = False

        def mutation(connection):
            nonlocal created
            self.ledger.lock_job_identity(
                request["job_id"],
                connection=connection,
            )
            durable_identity = self.ledger.get_job_identity(
                request["job_id"],
                connection=connection,
            )
            if durable_identity is not None:
                if (
                    durable_identity["owner_subject"] != owner
                    or durable_identity["create_hash"] != request_hash
                ):
                    raise conflict(
                        "jobId has already been used for a different request"
                    )
                return durable_identity["create_result"], request["job_id"]

            active = self.ledger.active_job_count(
                owner,
                connection=connection,
            )
            if active >= self.config.max_active_jobs_per_subject:
                raise ServiceError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "active job quota exceeded",
                )

            model_config = request.get("model_config")
            resolved_model_ref = None
            if request["operation"] == "predict":
                model = _resolve_model(
                    self.ledger,
                    owner,
                    request["model_selector"],
                    request["model_ref"],
                    connection=connection,
                )
                _verify_model_contract(model, request["data_contract"])
                self._model_validator(model)
                resolved_model_ref = model.model_ref
                model_config = _model_config_from_metadata(model.metadata)

            response = self._response(
                request["request_id"],
                jobId=request["job_id"],
                operation=request["operation"],
                revision=1,
                input={
                    "state": "OPEN",
                    "revision": 0,
                    "nextOrdinal": 0,
                },
                execution={"state": "WAITING_INPUT"},
                ownership={
                    "clientExecutionId": request["client_execution_id"],
                    "fencingToken": "1",
                },
                device={"requested": request["device"], "selected": None},
                resolvedModelRef=resolved_model_ref,
                dataContract=self._data_contract(request["data_contract"]),
                limits=self._limits(),
                upload={
                    "descriptorPath": [
                        "transformer",
                        self.contract.path_version,
                        "jobs",
                        request["job_id"],
                        "inputs",
                        "{ordinal}",
                    ],
                    "schemaId": (
                        self.contract.fit_schema_id
                        if request["operation"] == "fit"
                        else self.contract.predict_schema_id
                    ),
                    "oneDoPutIsOneSemanticPayload": True,
                },
            )
            self.ledger.create_job(
                job_id=request["job_id"],
                owner_subject=owner,
                client_execution_id=request["client_execution_id"],
                operation=request["operation"],
                requested_device=request["device"],
                prediction_column=request["prediction_column"],
                config_hash=request_hash,
                data_contract=request["data_contract"],
                create_result=response,
                model_label=request.get("model_label"),
                resolved_model_ref=resolved_model_ref,
                model_config=model_config,
                training_config=request.get("train_config"),
                connection=connection,
            )
            created = True
            return response, request["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=self.contract.create_action,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
        )
        if not replayed and created:
            self.metrics.add("jobsCreated")
            self.metrics.record_transition(
                "NONE",
                ExecutionState.WAITING_INPUT.value,
            )
            self.logger.event(
                "flight.job.created",
                requestId=request["request_id"],
                jobId=request["job_id"],
                operation=request["operation"],
                device=request["device"],
                inputState="OPEN",
                executionState=ExecutionState.WAITING_INPUT.value,
            )
        return response


class AcquireJobAction:
    """Transfer external ownership and advance the server fence."""

    def __init__(
        self,
        config,
        ledger,
        *,
        action_contract: JobActionContract,
        request_hasher: Callable[[dict], str],
        response_factory: Callable[..., dict],
        cleanup_candidate: Callable[[str, str], None],
        logger,
    ):
        self.config = config
        self.ledger = ledger
        self.contract = action_contract
        self._request_hash = request_hasher
        self._response = response_factory
        self._cleanup_candidate = cleanup_candidate
        self.logger = logger

    def acquire(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = self._request_hash(document)
        replay = _replay(
            self.ledger,
            owner,
            self.contract.acquire_action,
            request["idempotency_key"],
            request_hash,
        )
        if replay is not None:
            return replay
        cleanup: tuple[tuple[str, str], ...] = ()

        def mutation(connection):
            nonlocal cleanup
            job, cleanup = self.ledger.acquire_job(
                request["job_id"],
                owner_subject=owner,
                previous_client_execution_id=(
                    request["previous_client_execution_id"]
                ),
                expected_fencing_token=request["expected_fencing_token"],
                client_execution_id=request["client_execution_id"],
                acquire_grace_seconds=self.config.acquire_idle_grace_seconds,
                connection=connection,
            )
            return self._response(
                request["request_id"],
                jobId=job["job_id"],
                revision=job["revision"],
                ownership={
                    "clientExecutionId": job["client_execution_id"],
                    "fencingToken": str(job["fencing_token"]),
                },
            ), job["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=self.contract.acquire_action,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
        )
        if not replayed:
            for storage_class, relative_path in cleanup:
                self._cleanup_candidate(storage_class, relative_path)
            self.logger.event(
                "flight.job.acquired",
                requestId=request["request_id"],
                jobId=request["job_id"],
                clientExecutionId=request["client_execution_id"],
                fencingToken=response["ownership"]["fencingToken"],
            )
        return response


class InputCloseAction:
    """Commit EOF and the immutable complete input summary."""

    def __init__(
        self,
        ledger,
        *,
        cuda_available: Callable[[], bool],
        queue_notifier: Callable[[str], None],
        action_contract: JobActionContract,
        request_hasher: Callable[[dict], str],
        response_factory: Callable[..., dict],
        metrics,
        logger,
    ):
        self.ledger = ledger
        self._cuda_available = cuda_available
        self._queue_notifier = queue_notifier
        self.contract = action_contract
        self._request_hash = request_hasher
        self._response = response_factory
        self.metrics = metrics
        self.logger = logger

    def close(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = self._request_hash(document)
        queued = False

        def mutation(connection):
            nonlocal queued
            current = self.ledger.get_job(
                request["job_id"],
                owner_subject=owner,
                connection=connection,
                for_update=True,
            )
            if current is None:
                raise not_found("job not found")
            selected = current["selected_device"] or _select_device(
                current["requested_device"],
                self._cuda_available(),
            )
            job, repeated = self.ledger.close_input(
                request["job_id"],
                client_execution_id=request["client_execution_id"],
                fencing_token=request["fencing_token"],
                payload_count=request["payload_count"],
                total_rows=request["total_rows"],
                total_bytes=request["total_bytes"],
                manifest_sha256=request["manifest_sha256"],
                selected_device=selected,
                connection=connection,
            )
            queued = (
                not repeated
                and current["execution_state"]
                == ExecutionState.WAITING_INPUT.value
                and job["execution_state"] == ExecutionState.QUEUED.value
            )
            return self._response(
                request["request_id"],
                jobId=job["job_id"],
                revision=job["revision"],
                input={
                    "state": job["input_state"],
                    "revision": job["input_revision"],
                    "payloadCount": job["payload_count"],
                    "totalRows": job["total_rows"],
                    "totalBytes": job["total_bytes"],
                    "manifestSha256": job["manifest_sha256"],
                },
                execution={"state": job["execution_state"]},
            ), job["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=self.contract.input_close_action,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
            replay_guard=lambda connection: _require_current_fence(
                self.ledger,
                owner,
                request,
                connection,
            ),
        )
        if not replayed:
            self.metrics.add("inputsClosed")
            self.logger.event(
                "flight.input.closed",
                requestId=request["request_id"],
                jobId=request["job_id"],
                payloadCount=request["payload_count"],
                totalRows=request["total_rows"],
            )
            if queued:
                self._queue_notifier(request["job_id"])
        return response


class CancelJobAction:
    """Cancel under the current external ownership fence."""

    def __init__(
        self,
        ledger,
        *,
        cancel_notifier: Callable[[str], None],
        cleanup_candidate: Callable[[str, str], None],
        action_contract: JobActionContract,
        request_hasher: Callable[[dict], str],
        response_factory: Callable[..., dict],
        metrics,
        logger,
    ):
        self.ledger = ledger
        self._cancel_notifier = cancel_notifier
        self._cleanup_candidate = cleanup_candidate
        self.contract = action_contract
        self._request_hash = request_hasher
        self._response = response_factory
        self.metrics = metrics
        self.logger = logger

    def cancel(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = self._request_hash(document)
        notify = False
        cleanup: tuple[tuple[str, str], ...] = ()

        def mutation(connection):
            nonlocal notify, cleanup
            job, notify, cleanup = self.ledger.cancel_job(
                request["job_id"],
                owner_subject=owner,
                client_execution_id=request["client_execution_id"],
                fencing_token=request["fencing_token"],
                connection=connection,
            )
            return self._response(
                request["request_id"],
                jobId=job["job_id"],
                revision=job["revision"],
                input={"state": job["input_state"]},
                execution={"state": job["execution_state"]},
            ), job["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=self.contract.cancel_action,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
            replay_guard=lambda connection: _require_current_fence(
                self.ledger,
                owner,
                request,
                connection,
            ),
        )
        if not replayed:
            for storage_class, relative_path in cleanup:
                self._cleanup_candidate(storage_class, relative_path)
            self.metrics.add("jobsCancellationRequested")
            self.logger.event(
                "flight.job.cancelled",
                requestId=request["request_id"],
                jobId=request["job_id"],
                inputState=response["input"]["state"],
                executionState=response["execution"]["state"],
            )
            if notify:
                self._cancel_notifier(request["job_id"])
        return response


def _replay(
    ledger,
    owner: str,
    action: str,
    idempotency_key: str,
    request_hash: str,
) -> dict | None:
    record = ledger.lookup_idempotency(owner, action, idempotency_key)
    if record is None:
        return None
    if record["request_hash"] != request_hash:
        raise conflict("idempotency key was used for a different request")
    return record["response"]


def _require_current_fence(ledger, owner, request, connection) -> None:
    job = ledger.get_job(
        request["job_id"],
        owner_subject=owner,
        connection=connection,
        for_update=True,
    )
    if job is None:
        raise not_found("job not found")
    if (
        job["client_execution_id"] != request["client_execution_id"]
        or job["fencing_token"] != request["fencing_token"]
    ):
        raise ServiceError(
            ErrorCode.STALE_FENCE,
            "job ownership fence is stale",
        )


def _resolve_model(
    ledger,
    owner: str,
    selector: str,
    value: str,
    *,
    connection,
):
    if selector == "modelAlias":
        model = ledger.resolve_published_model_alias(
            owner,
            value,
            connection=connection,
        )
    else:
        model = ledger.get_published_model(
            value,
            owner_subject=owner,
            connection=connection,
        )
    if model is None:
        raise not_found("model generation not found")
    return model


def _verify_model_contract(model, data_contract: dict) -> None:
    if not model.certified_for_v3 or model.data_contract is None:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model generation is not certified for Flight v3",
        )
    if (
        model.data_contract.get("data_contract_sha256")
        != data_contract["data_contract_sha256"]
    ):
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model data contract does not match prediction input",
        )


def _model_config_from_metadata(metadata: dict):
    from app.contracts.worker.v2.config import ModelConfig

    try:
        config = ModelConfig.from_dict(metadata.get("model_config"))
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model metadata is invalid",
        ) from exc
    if config is None:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model configuration is unavailable",
        )
    return config


def _select_device(requested: str, cuda_available: bool) -> str:
    decision = resolve_device(requested, cuda_available)
    if decision.error_code is not None:
        raise ServiceError(
            decision.error_code,
            "explicit CUDA device is unavailable",
        )
    if decision.selected is None:
        raise ServiceError(ErrorCode.INTERNAL, "device selection failed")
    return decision.selected


__all__ = [
    "AcquireJobAction",
    "CancelJobAction",
    "CreateJobAction",
    "InputCloseAction",
    "JobActionContract",
]
