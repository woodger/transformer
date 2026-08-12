from __future__ import annotations

from collections.abc import Callable

from app.service.application.job_models import (
    AcquireJobCommand,
    CancelJobCommand,
    CloseInputCommand,
    CreateJobCommand,
    InputClosed,
    JobAcquired,
    JobCancelled,
    JobCreated,
    JobCreationPreparation,
    ServiceLimits,
)
from app.service.application.ports.job_lifecycle import (
    CandidateArtifactCleaner,
    JobLifecycleStore,
    ModelArtifactVerifier,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.policies import resolve_device


class CreateJobAction:
    """Create a client-identified durable job in one transaction."""

    def __init__(
        self,
        store: JobLifecycleStore,
        *,
        max_active_jobs: int,
        limits: ServiceLimits,
        cuda_available: Callable[[], bool],
        is_draining: Callable[[], bool],
        model_verifier: ModelArtifactVerifier,
        metrics,
        logger,
    ):
        self.store = store
        self.max_active_jobs = max_active_jobs
        self.limits = limits
        self._cuda_available = cuda_available
        self._is_draining = is_draining
        self._model_verifier = model_verifier
        self.metrics = metrics
        self.logger = logger

    def create(self, command: CreateJobCommand) -> JobCreated:
        def preflight() -> None:
            if self._is_draining():
                raise ServiceError(ErrorCode.UNAVAILABLE, "service is draining")
            if (
                command.requested_device == "cuda"
                and not self._cuda_available()
            ):
                self.metrics.add("cudaUnavailableRequests")
                raise ServiceError(
                    ErrorCode.DEVICE_UNAVAILABLE,
                    "explicit CUDA device is not available",
                )

        def prepare(model) -> JobCreationPreparation:
            model_config = command.model_config
            resolved_model_ref = None
            if command.operation == "predict":
                if model is None:
                    raise ServiceError(
                        ErrorCode.NOT_FOUND,
                        "model generation not found",
                    )
                _verify_model_contract(model, command.data_contract)
                self._model_verifier.verify(model)
                resolved_model_ref = model.model_ref
                model_config = _model_config_from_metadata(model.metadata)
            if model_config is None:
                raise ServiceError(
                    ErrorCode.INVALID_ARGUMENT,
                    "job model configuration is unavailable",
                )

            result = JobCreated(
                request_id=command.request_id,
                job_id=command.job_id,
                operation=command.operation,
                revision=1,
                input_state=InputState.OPEN,
                input_revision=0,
                next_input_ordinal=0,
                execution_state=ExecutionState.WAITING_INPUT,
                client_execution_id=command.client_execution_id,
                fencing_token=1,
                requested_device=command.requested_device,
                selected_device=None,
                resolved_model_ref=resolved_model_ref,
                data_contract=dict(command.data_contract),
                limits=self.limits,
            )
            return JobCreationPreparation(
                result=result,
                model_config=model_config,
                training_config=command.training_config,
                resolved_model_ref=resolved_model_ref,
            )

        outcome = self.store.create(
            command,
            max_active_jobs=self.max_active_jobs,
            preflight=preflight,
            prepare=prepare,
        )
        if not outcome.replayed:
            self.metrics.add("jobsCreated")
            self.metrics.record_transition(
                "NONE",
                ExecutionState.WAITING_INPUT.value,
            )
            self.logger.event(
                "flight.job.created",
                requestId=outcome.result.request_id,
                jobId=outcome.result.job_id,
                operation=outcome.result.operation,
                device=outcome.result.requested_device,
                inputState=outcome.result.input_state.value,
                executionState=outcome.result.execution_state.value,
            )
        return outcome.result


class AcquireJobAction:
    """Transfer external ownership and advance the server fence."""

    def __init__(
        self,
        store: JobLifecycleStore,
        *,
        acquire_grace_seconds: float,
        artifact_cleaner: CandidateArtifactCleaner,
        logger,
    ):
        self.store = store
        self.acquire_grace_seconds = acquire_grace_seconds
        self._artifact_cleaner = artifact_cleaner
        self.logger = logger

    def acquire(self, command: AcquireJobCommand) -> JobAcquired:
        outcome = self.store.acquire(
            command,
            acquire_grace_seconds=self.acquire_grace_seconds,
        )
        if not outcome.replayed:
            for location in outcome.cleanup:
                self._artifact_cleaner.cleanup(location)
            self.logger.event(
                "flight.job.acquired",
                requestId=outcome.result.request_id,
                jobId=outcome.result.job_id,
                clientExecutionId=outcome.result.client_execution_id,
                fencingToken=str(outcome.result.fencing_token),
            )
        return outcome.result


class InputCloseAction:
    """Commit EOF and the immutable complete input summary."""

    def __init__(
        self,
        store: JobLifecycleStore,
        *,
        cuda_available: Callable[[], bool],
        queue_notifier: Callable[[str], None],
        metrics,
        logger,
    ):
        self.store = store
        self._cuda_available = cuda_available
        self._queue_notifier = queue_notifier
        self.metrics = metrics
        self.logger = logger

    def close(self, command: CloseInputCommand) -> InputClosed:
        def select_device(
            requested: str,
            selected: str | None,
            operation: str,
            total_rows: int,
        ) -> str:
            if operation == "fit" and total_rows == 0:
                return selected or "cpu"
            return selected or _select_device(
                requested,
                self._cuda_available(),
            )

        outcome = self.store.close_input(
            command,
            select_device=select_device,
        )
        if not outcome.replayed:
            self.metrics.add("inputsClosed")
            self.logger.event(
                "flight.input.closed",
                requestId=outcome.result.request_id,
                jobId=outcome.result.job_id,
                payloadCount=outcome.result.payload_count,
                totalRows=outcome.result.total_rows,
            )
            if outcome.queued:
                self._queue_notifier(outcome.result.job_id)
        return outcome.result


class CancelJobAction:
    """Cancel under the current external ownership fence."""

    def __init__(
        self,
        store: JobLifecycleStore,
        *,
        cancel_notifier: Callable[[str], None],
        artifact_cleaner: CandidateArtifactCleaner,
        metrics,
        logger,
    ):
        self.store = store
        self._cancel_notifier = cancel_notifier
        self._artifact_cleaner = artifact_cleaner
        self.metrics = metrics
        self.logger = logger

    def cancel(self, command: CancelJobCommand) -> JobCancelled:
        outcome = self.store.cancel(command)
        if not outcome.replayed:
            for location in outcome.cleanup:
                self._artifact_cleaner.cleanup(location)
            self.metrics.add("jobsCancellationRequested")
            self.logger.event(
                "flight.job.cancelled",
                requestId=outcome.result.request_id,
                jobId=outcome.result.job_id,
                inputState=outcome.result.input_state.value,
                executionState=outcome.result.execution_state.value,
            )
            if outcome.notify_worker:
                self._cancel_notifier(outcome.result.job_id)
        return outcome.result


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
]
