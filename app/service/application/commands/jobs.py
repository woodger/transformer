from __future__ import annotations

from collections.abc import Callable
from typing import cast

from app.service.application.messages.jobs import (
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
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.services.model_contract import (
    verify_model_for_predict,
    verify_parent_model_for_fit,
)
from app.service.domain.errors import ServiceError
from app.service.domain.initialization import (
    published_model_initialization,
    random_initialization,
)
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.json_types import JsonObject
from app.service.domain.policies import resolve_device
from app.service.domain.records import PublishedModelRecord


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
        job_config_digest: Callable[[JsonObject], str],
        model_verifier: ModelArtifactVerifier,
        metrics: OperationalMetricSink,
        logger: EventLogger,
    ) -> None:
        self.store = store
        self.max_active_jobs = max_active_jobs
        self.limits = limits
        self._cuda_available = cuda_available
        self._is_draining = is_draining
        self._job_config_digest = job_config_digest
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
                self.metrics.add("gpuUnavailableRequests")
                raise ServiceError(
                    ErrorCode.DEVICE_UNAVAILABLE,
                    "explicit GPU device is not available",
                )

        def prepare(
            model: PublishedModelRecord | None,
        ) -> JobCreationPreparation:
            model_config = command.model_config
            resolved_model_ref = None
            initialization = None
            if command.operation == "predict":
                if model is None:
                    raise ServiceError(
                        ErrorCode.NOT_FOUND,
                        "model generation not found",
                    )
                model_config = verify_model_for_predict(
                    model,
                    model_contract=command.model_contract,
                    semantic_digests=command.semantic_digests,
                )
                self._model_verifier.verify(model)
                resolved_model_ref = model.model_ref
            elif command.initialization_source == "random":
                initialization = random_initialization()
            elif command.initialization_source == "publishedModel":
                if model is None:
                    raise ServiceError(
                        ErrorCode.NOT_FOUND,
                        "model generation not found",
                    )
                model_config = verify_parent_model_for_fit(
                    model,
                    model_config=model_config,
                    model_contract=command.model_contract,
                    semantic_digests=command.semantic_digests,
                )
                self._model_verifier.verify(model)
                resolved_model_ref = model.model_ref
                initialization = published_model_initialization(
                    model.model_ref,
                    model.sha256,
                    model.semantic_digests,
                    command.semantic_digests,
                )
            else:
                raise ServiceError(
                    ErrorCode.INVALID_ARGUMENT,
                    "fit initialization is unavailable",
                )
            if model_config is None:
                raise ServiceError(
                    ErrorCode.INVALID_ARGUMENT,
                    "job model configuration is unavailable",
                )

            config_document = _job_config_document(
                command,
                initialization=initialization,
                resolved_model_ref=resolved_model_ref,
            )
            config_digest = self._job_config_digest(config_document)

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
                source_encoding=dict(command.source_encoding),
                data_contract=dict(command.data_contract),
                model_contract=dict(command.model_contract),
                semantic_digests=dict(command.semantic_digests),
                job_config_sha256=config_digest,
                limits=self.limits,
                initialization=initialization,
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


def _job_config_document(
    command: CreateJobCommand,
    *,
    initialization: JsonObject | None,
    resolved_model_ref: str | None,
) -> JsonObject:
    common: JsonObject = {
        "operation": command.operation,
        "requestedDevice": (
            "gpu" if command.requested_device == "cuda" else command.requested_device
        ),
        "sourceEncoding": dict(command.source_encoding),
        "dataContractSha256": cast(
            str,
            command.semantic_digests["dataContractSha256"],
        ),
        "modelContractSha256": cast(
            str,
            command.semantic_digests["modelContractSha256"],
        ),
    }
    if command.operation == "predict":
        if resolved_model_ref is None:
            raise AssertionError("predict model reference is unresolved")
        common.update({
            "resolvedModelRef": resolved_model_ref,
            "predictionColumn": command.prediction_column,
        })
        return common

    if (
        command.model_label is None
        or command.training_config is None
        or initialization is None
    ):
        raise AssertionError("fit job configuration is unresolved")
    common.update({
        "modelLabel": command.model_label,
        "trainingConfig": command.training_config.to_manifest(),
        "diagnostics": command.training_config.diagnostics.to_document(),
        "initialization": initialization,
    })
    return common


class AcquireJobAction:
    """Transfer external ownership and advance the server fence."""

    def __init__(
        self,
        store: JobLifecycleStore,
        *,
        acquire_grace_seconds: float,
        artifact_cleaner: CandidateArtifactCleaner,
        logger: EventLogger,
    ) -> None:
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
        metrics: OperationalMetricSink,
        logger: EventLogger,
    ) -> None:
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
                totalChunks=outcome.result.total_chunks,
                totalLogicalRows=outcome.result.total_rows,
                totalNativeRows=list(outcome.result.total_native_rows),
                rangeCount=outcome.result.range_count,
                totalBytes=outcome.result.total_bytes,
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
        metrics: OperationalMetricSink,
        logger: EventLogger,
    ) -> None:
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


def _select_device(requested: str, cuda_available: bool) -> str:
    decision = resolve_device(requested, cuda_available)
    if decision.error_code is not None:
        raise ServiceError(
            decision.error_code,
            "explicit GPU device is unavailable",
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
