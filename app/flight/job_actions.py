from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from app.flight.constants import (
    CANCEL_ACTION,
    CONTRACT_PATH_VERSION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
    SEAL_ACTION,
    START_ACTION,
    TERMINAL_STATES,
    ErrorCode,
    JobState,
)
from app.flight.contract import (
    canonical_manifest_hash,
    canonical_request_hash,
    response_document,
)
from app.flight.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
)
from app.flight.state import decide_cancel
from app.training.run_config import ModelConfig


class CreateStatusActions:
    """Create jobs and render their durable public status."""

    def __init__(
        self,
        config,
        ledger,
        *,
        cuda_available: Callable[[], bool],
        is_draining: Callable[[], bool],
        limits: Callable[[], dict],
        metrics,
        logger,
    ):
        self.config = config
        self.ledger = ledger
        self._cuda_available = cuda_available
        self._is_draining = is_draining
        self._limits = limits
        self.metrics = metrics
        self.logger = logger

    def create(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = canonical_request_hash(document)
        replay = _replay(
            self.ledger,
            owner,
            CREATE_ACTION,
            request["idempotency_key"],
            request_hash,
        )
        if replay is not None:
            return replay
        if self._is_draining():
            raise ServiceError(ErrorCode.UNAVAILABLE, "service is draining")
        if request["device"] == "cuda" and not self._cuda_available():
            self.metrics.add("cudaUnavailableRequests")
            raise ServiceError(
                ErrorCode.DEVICE_UNAVAILABLE,
                "explicit CUDA device is not available",
            )

        model_config = request.get("model_config")
        input_model_ref = None
        if request["operation"] == "predict":
            if request["model_selector"] == "modelAlias":
                model = self.ledger.resolve_model_alias(
                    owner,
                    request["model_ref"],
                )
            else:
                model = self.ledger.get_model(
                    request["model_ref"],
                    owner_subject=owner,
                )
            if model is None:
                raise not_found("model generation not found")
            input_model_ref = model["model_ref"]
            model_config = _model_config_from_metadata(model["metadata"])

        def mutation(connection):
            job_id = str(uuid.uuid4())
            active = self.ledger.active_job_count(
                owner,
                connection=connection,
            )
            if active >= self.config.max_active_jobs_per_subject:
                raise ServiceError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "active job quota exceeded",
                )
            job = self.ledger.create_job(
                job_id=job_id,
                owner_subject=owner,
                operation=request["operation"],
                requested_device=request["device"],
                prediction_column=request["prediction_column"],
                config_hash=request_hash,
                model_label=request.get("model_label"),
                input_model_ref=input_model_ref,
                model_config=model_config,
                training_config=request.get("train_config"),
                connection=connection,
            )
            response = response_document(
                request["request_id"],
                jobId=job_id,
                operation=job["operation"],
                state=job["state"],
                revision=job["revision"],
                device={
                    "requested": job["requested_device"],
                    "selected": None,
                },
                resolvedModelRef=input_model_ref,
                limits=self._limits(),
                upload={
                    "descriptorPath": [
                        "transformer",
                        CONTRACT_PATH_VERSION,
                        "jobs",
                        job_id,
                        "inputs",
                        "{ordinal}",
                    ],
                    "schemaId": (
                        FIT_SCHEMA_ID
                        if job["operation"] == "fit"
                        else PREDICT_SCHEMA_ID
                    ),
                    "oneDoPutIsOneSemanticFrame": True,
                },
            )
            return response, job_id

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=CREATE_ACTION,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
        )
        if not replayed:
            self.metrics.add("jobsCreated")
            self.metrics.record_transition(
                "NONE",
                JobState.UPLOADING.value,
            )
            self.logger.event(
                "flight.job.created",
                requestId=request["request_id"],
                jobId=response["jobId"],
                operation=request["operation"],
                device=request["device"],
                state=JobState.UPLOADING.value,
                fromState=None,
                toState=JobState.UPLOADING.value,
            )
        return response

    def status(self, owner: str, job_id: str, request_id: str) -> dict:
        job, inputs, outputs, recovery = (
            self.ledger.get_status_snapshot(job_id, owner)
        )
        if job is None:
            raise not_found("job not found")
        terminal = JobState(job["state"]) in TERMINAL_STATES
        result = job.get("result") or {}
        # A cancellation is a terminal state, not an execution error.
        error = None
        if job["state"] == JobState.FAILED.value:
            error = {
                "code": job["error_code"],
                "message": job["error_message"],
            }
        return response_document(
            request_id,
            jobId=job_id,
            operation=job["operation"],
            state=job["state"],
            revision=job["revision"],
            timestamps=_timestamps(job),
            device={
                "requested": job["requested_device"],
                "selected": job["selected_device"],
            },
            committedInputs=[_safe_input(item) for item in inputs],
            progress=job.get("progress") or {},
            attempt=job["attempt"],
            recovery=_safe_recovery(recovery),
            error=error,
            results={
                "outputs": [
                    {
                        "ordinal": item["ordinal"],
                        "descriptorPath": [
                            "transformer",
                            CONTRACT_PATH_VERSION,
                            "jobs",
                            job_id,
                            "outputs",
                            str(item["ordinal"]),
                        ],
                        "rows": item["rows"],
                        "bytes": item["bytes"],
                    }
                    for item in outputs
                ],
                "modelRef": result.get("modelRef"),
                "checkpoint": result.get("checkpoint"),
            },
            pollAfterMs=0 if terminal else 500,
        )


class LifecycleActions:
    """Seal, queue, and cancel durable jobs."""

    def __init__(
        self,
        ledger,
        *,
        cuda_available: Callable[[], bool],
        is_draining: Callable[[], bool],
        queue_notifier: Callable[[str], None],
        cancel_notifier: Callable[[str], None],
        metrics,
        logger,
    ):
        self.ledger = ledger
        self._cuda_available = cuda_available
        self._is_draining = is_draining
        self._queue_notifier = queue_notifier
        self._cancel_notifier = cancel_notifier
        self.metrics = metrics
        self.logger = logger

    def seal(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = canonical_request_hash(document)
        replay = _replay(
            self.ledger,
            owner,
            SEAL_ACTION,
            request["idempotency_key"],
            request_hash,
        )
        if replay is not None:
            return replay
        job = _owned_job(self.ledger, owner, request["job_id"])
        manifest = request["manifest"]
        transitioned = False

        def mutation(connection):
            nonlocal transitioned
            inputs = self.ledger.list_inputs(
                job["job_id"],
                connection=connection,
            )
            source_width, feature_dim = _validate_manifest(
                job,
                manifest,
                inputs,
            )
            manifest_hash = canonical_manifest_hash(manifest)
            current = self.ledger.get_job(
                job["job_id"],
                connection=connection,
                for_update=True,
            )
            first_response = response_document(
                request["request_id"],
                jobId=job["job_id"],
                state=JobState.SEALED.value,
                revision=current["revision"] + 1,
                manifestSha256=manifest_hash,
                inputs=len(inputs),
            )
            sealed_job, repeated = self.ledger.seal_job(
                job["job_id"],
                manifest_hash=manifest_hash,
                manifest=manifest,
                source_width=source_width,
                feature_dim=feature_dim,
                result=first_response,
                connection=connection,
            )
            transitioned = not repeated
            if repeated:
                stored = sealed_job.get("seal_result")
                if stored is None:
                    # Compatibility for ledgers produced before lifecycle
                    # snapshots were populated. Each committed input and the
                    # seal transition increment revision exactly once.
                    stored = response_document(
                        request["request_id"],
                        jobId=job["job_id"],
                        state=JobState.SEALED.value,
                        revision=len(inputs) + 2,
                        manifestSha256=manifest_hash,
                        inputs=len(inputs),
                    )
                response = {
                    **stored,
                    "requestId": request["request_id"],
                }
            else:
                response = first_response
            return response, job["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=SEAL_ACTION,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
        )
        if not replayed and transitioned:
            self.metrics.add("jobsSealed")
            self.metrics.record_transition(
                JobState.UPLOADING.value,
                JobState.SEALED.value,
            )
            self.logger.event(
                "flight.job.transition",
                requestId=request["request_id"],
                jobId=job["job_id"],
                fromState=JobState.UPLOADING.value,
                toState=JobState.SEALED.value,
            )
        return response

    def start(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = canonical_request_hash(document)
        replay = _replay(
            self.ledger,
            owner,
            START_ACTION,
            request["idempotency_key"],
            request_hash,
        )
        if replay is not None:
            return replay
        job = _owned_job(self.ledger, owner, request["job_id"])
        transitioned = False

        def mutation(connection):
            nonlocal transitioned
            current = self.ledger.get_job(
                job["job_id"],
                connection=connection,
                for_update=True,
            )
            if self._is_draining() and current["queued_at"] is None:
                raise ServiceError(
                    ErrorCode.UNAVAILABLE,
                    "service is draining",
                )
            if current["selected_device"]:
                selected = current["selected_device"]
            else:
                selected = self._select_device(
                    current["requested_device"],
                )
            first_response = response_document(
                request["request_id"],
                jobId=job["job_id"],
                state=JobState.QUEUED.value,
                revision=current["revision"] + 1,
                device={
                    "requested": current["requested_device"],
                    "selected": selected,
                },
            )
            queued, repeated = self.ledger.queue_job(
                job["job_id"],
                selected_device=selected,
                result=first_response,
                connection=connection,
            )
            transitioned = not repeated
            if repeated:
                stored = queued.get("start_result")
                if stored is None:
                    input_count = len(
                        self.ledger.list_inputs(
                            job["job_id"],
                            connection=connection,
                        )
                    )
                    stored = response_document(
                        request["request_id"],
                        jobId=job["job_id"],
                        state=JobState.QUEUED.value,
                        revision=input_count + 3,
                        device={
                            "requested": queued["requested_device"],
                            "selected": queued["selected_device"],
                        },
                    )
                response = {
                    **stored,
                    "requestId": request["request_id"],
                }
            else:
                response = first_response
            return response, job["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=START_ACTION,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
        )
        if not replayed and transitioned:
            self.metrics.add("jobsQueued")
            self.metrics.record_transition(
                JobState.SEALED.value,
                JobState.QUEUED.value,
            )
            self.logger.event(
                "flight.job.transition",
                requestId=request["request_id"],
                jobId=job["job_id"],
                fromState=JobState.SEALED.value,
                toState=JobState.QUEUED.value,
            )
            self._queue_notifier(job["job_id"])
        return response

    def cancel(self, owner: str, request: dict, document: dict) -> dict:
        request_hash = canonical_request_hash(document)
        replay = _replay(
            self.ledger,
            owner,
            CANCEL_ACTION,
            request["idempotency_key"],
            request_hash,
        )
        if replay is not None:
            return replay
        job = _owned_job(self.ledger, owner, request["job_id"])
        notify = False
        transition = None

        def mutation(connection):
            nonlocal notify, transition
            current_row = self.ledger.get_job(
                job["job_id"],
                connection=connection,
                for_update=True,
            )
            current = JobState(current_row["state"])
            decision = decide_cancel(current)
            if decision.target is None:
                changed = dict(current_row)
            elif decision.target == JobState.CANCELLED:
                changed = self.ledger.transition_job(
                    job["job_id"],
                    decision.target,
                    updates={
                        "finished_at": time.time(),
                        "cancel_requested_at": time.time(),
                    },
                    connection=connection,
                )
                transition = (current.value, decision.target.value)
            elif decision.target == JobState.CANCELLING:
                changed = self.ledger.transition_job(
                    job["job_id"],
                    decision.target,
                    updates={"cancel_requested_at": time.time()},
                    connection=connection,
                )
                notify = decision.notify_worker
                transition = (current.value, decision.target.value)
            else:
                raise failed_precondition("job cannot be cancelled")
            response = response_document(
                request["request_id"],
                jobId=job["job_id"],
                state=changed["state"],
                revision=changed["revision"],
            )
            return response, job["job_id"]

        response, replayed = self.ledger.run_idempotent(
            owner_subject=owner,
            action_name=CANCEL_ACTION,
            idempotency_key=request["idempotency_key"],
            request_hash=request_hash,
            mutation=mutation,
        )
        if not replayed and transition is not None:
            self.metrics.add("jobsCancellationRequested")
            self.metrics.record_transition(*transition)
            self.logger.event(
                "flight.job.transition",
                requestId=request["request_id"],
                jobId=job["job_id"],
                fromState=transition[0],
                toState=transition[1],
            )
        if notify and not replayed:
            self._cancel_notifier(job["job_id"])
        return response

    def _select_device(self, requested: str) -> str:
        available = bool(self._cuda_available())
        if requested == "cuda":
            if not available:
                self.metrics.add("cudaUnavailableRequests")
                raise ServiceError(
                    ErrorCode.DEVICE_UNAVAILABLE,
                    "explicit CUDA device became unavailable before start",
                )
            return "cuda"
        if requested == "auto":
            return "cuda" if available else "cpu"
        return "cpu"


def _safe_recovery(recovery) -> dict | None:
    if recovery is None:
        return None
    checkpoint = recovery.checkpoint
    return {
        "latestCheckpoint": (
            None
            if checkpoint is None
            else {
                "generation": checkpoint.generation,
                "completedEpochs": checkpoint.completed_epochs,
                "globalStep": checkpoint.global_step,
                "trainingComplete": checkpoint.training_complete,
            }
        ),
        "resumedFromGeneration": recovery.resumed_from_generation,
        "retryCount": recovery.retry_count,
        "lastRetryCode": recovery.last_retry_code,
        "boundary": "globalEpoch",
    }


def _replay(
    ledger,
    owner: str,
    action: str,
    idempotency_key: str,
    request_hash: str,
) -> dict | None:
    record = ledger.lookup_idempotency(
        owner,
        action,
        idempotency_key,
    )
    if record is None:
        return None
    if record["request_hash"] != request_hash:
        raise conflict(
            "idempotency key was used for a different request"
        )
    return record["response"]


def _owned_job(ledger, owner: str, job_id: str) -> dict:
    job = ledger.get_job(job_id, owner_subject=owner)
    if job is None:
        raise not_found("job not found")
    return job


def _validate_manifest(
    job: dict,
    manifest: list[dict],
    inputs: list[dict],
) -> tuple[int | None, int | None]:
    ordinals = [item["ordinal"] for item in manifest]
    if ordinals != list(range(len(manifest))):
        raise failed_precondition(
            "manifest ordinals must be ordered and contiguous from zero"
        )
    if len(inputs) != len(manifest):
        raise failed_precondition(
            "manifest does not match the committed input set"
        )
    expected_schema = (
        FIT_SCHEMA_ID
        if job["operation"] == "fit"
        else PREDICT_SCHEMA_ID
    )
    fingerprints = set()
    widths = set()
    for expected, actual in zip(manifest, inputs, strict=True):
        if (
            expected["ordinal"] != actual["ordinal"]
            or expected["payloadId"] != actual["payload_id"]
            or expected["sha256"] != actual["sha256"]
        ):
            raise failed_precondition(
                "manifest entry does not match committed input"
            )
        if actual["schema_id"] != expected_schema:
            raise failed_precondition(
                "input schemaId does not match job operation"
            )
        fingerprints.add(actual["schema_fingerprint"])
        if actual["source_width"] is not None:
            widths.add(actual["source_width"])
    if len(fingerprints) > 1:
        raise failed_precondition("input schemas are inconsistent")
    if len(widths) > 1:
        raise failed_precondition("input dimensions are inconsistent")
    source_width = next(iter(widths), None)
    model_config = ModelConfig.from_dict(job["model_config"])
    feature_dim = (
        None
        if source_width is None
        else source_width // model_config.seq_len
    )
    if (
        model_config.feature_dim is not None
        and feature_dim is not None
        and feature_dim != model_config.feature_dim
    ):
        raise failed_precondition(
            "input feature dimension conflicts with model"
        )
    return source_width, feature_dim


def _model_config_from_metadata(metadata: dict) -> ModelConfig:
    value = metadata.get("model_config") or metadata.get("modelConfig")
    if value is None:
        checkpoint = metadata.get("checkpoint") or {}
        value = checkpoint.get("model_config")
    config = ModelConfig.from_dict(value)
    if config is None or config.feature_dim is None:
        raise failed_precondition(
            "model generation does not contain a complete model configuration"
        )
    return config


def _safe_input(item: dict) -> dict:
    return {
        "payloadId": item["payload_id"],
        "ordinal": item["ordinal"],
        "schemaId": item["schema_id"],
        "rows": item["rows"],
        "batches": item["batches"],
        "bytes": item["bytes"],
        "sha256": item["sha256"],
        "schemaFingerprint": item["schema_fingerprint"],
    }


def _timestamps(job: dict) -> dict:
    return {
        "createdAt": _timestamp(job.get("created_at")),
        "updatedAt": _timestamp(job.get("updated_at")),
        "sealedAt": _timestamp(job.get("sealed_at")),
        "queuedAt": _timestamp(job.get("queued_at")),
        "startedAt": _timestamp(job.get("started_at")),
        "cancelRequestedAt": _timestamp(
            job.get("cancel_requested_at")
        ),
        "finishedAt": _timestamp(job.get("finished_at")),
    }


def _timestamp(value) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
