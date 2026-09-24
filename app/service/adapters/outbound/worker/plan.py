from __future__ import annotations

import hashlib
import os
from typing import Protocol

from app.contracts.json_types import JsonObject
from app.contracts.worker.v15 import (
    CHECKPOINT_FORMAT,
    CONTRACT_NAME,
    CONTRACT_VERSION,
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    RECOVERY_FORMAT,
    validate_document,
)
from app.contracts.worker.v15.config import train_config_to_manifest
from app.service.application.ports.jobs import JobRepository
from app.service.application.ports.workers import ExecutionInput, ExecutionPlan
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.initialization import validate_initialization
from app.service.domain.job import ErrorCode, InputState
from app.service.domain.records import (
    ExecutionJobRecord,
    ModelArtifactRecord,
    TrainingRecoveryCheckpointRecord,
)

_COPY_CHUNK_BYTES = 1024 * 1024


class WorkerPlanError(AttemptExecutionError):
    pass


class _PlanConfig(Protocol):
    @property
    def max_payload_bytes(self) -> int: ...


class _PlanLedger(JobRepository, Protocol):
    def get_model_artifact(
        self,
        model_ref: str,
        *,
        owner_subject: str,
    ) -> ModelArtifactRecord | None: ...


class _InputStore(Protocol):
    def absolute_path(self, relative_path: object) -> str: ...

    def input_directory(self, job_id: str) -> str: ...


class _PlanSpool(_InputStore, Protocol):
    def attempt_directory(self, job_id: str, attempt: int) -> str: ...

    def attempt_manifest_path(self, job_id: str, attempt: int) -> str: ...

    def ensure_parent(self, path: str) -> None: ...

    def model_absolute_path(self, relative_path: object) -> str: ...

    def model_checkpoint_path(self, model_ref: str) -> str: ...

    def write_json_once(
        self,
        destination: str,
        document: JsonObject,
    ) -> str: ...


class _RecoveryStore(_InputStore, Protocol):
    def checkpoint_path(self, job_id: str, generation: int) -> str: ...


class WorkerPlanBuilder:
    """Validate durable artifacts and render one trusted CLI execution plan."""

    def __init__(
        self,
        config: _PlanConfig,
        ledger: _PlanLedger,
        spool: _PlanSpool,
        recovery_store: _RecoveryStore | None = None,
        *,
        python_executable: str,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.recovery_store = recovery_store
        self.python_executable = python_executable

    def build(
        self,
        job: ExecutionJobRecord,
        attempt: int,
    ) -> ExecutionPlan:
        if attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        inputs = self._validated_inputs(job)
        recovery_checkpoint = (
            self.ledger.latest_recovery_checkpoint(job.job_id)
            if (
                job.operation == "fit"
                and self.recovery_store is not None
            )
            else None
        )
        manifest_path, workspace = self._write_worker_manifest(
            job,
            attempt,
            inputs,
            recovery_checkpoint,
        )
        argv = self._worker_argv(job, attempt, manifest_path)
        return ExecutionPlan(
            inputs=inputs,
            argv=argv,
            protocol_version=CONTRACT_VERSION,
            manifest_path=manifest_path,
            workspace=workspace,
        )

    def _worker_argv(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        manifest_path: str,
    ) -> tuple[str, ...]:
        if job.attempt_id is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "claimed job has no attempt identity",
            )
        return (
            self.python_executable,
            "-m",
            "app.worker.bootstrap",
            "run",
            f"--contract-version={CONTRACT_VERSION}",
            f"--job-id={job.job_id}",
            f"--attempt={attempt}",
            f"--attempt-id={job.attempt_id}",
            f"--manifest={manifest_path}",
        )

    def _write_worker_manifest(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        inputs: tuple[ExecutionInput, ...],
        recovery_checkpoint: TrainingRecoveryCheckpointRecord | None,
    ) -> tuple[str, str]:
        if job.attempt_id is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "claimed job has no attempt identity",
            )
        workspace = self.spool.attempt_directory(job.job_id, attempt)
        manifest_path = self.spool.attempt_manifest_path(job.job_id, attempt)
        self.spool.ensure_parent(manifest_path)
        device = job.selected_device
        if device not in ("cpu", "cuda"):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "queued job has no selected device",
            )
        if device == "cuda" and not job.assigned_device_id:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "GPU attempt has no assigned physical device",
            )
        if job.model_config is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "claimed job has no model configuration",
            )
        model_manifest: JsonObject = {}
        document: JsonObject = {
            "contract": CONTRACT_NAME,
            "protocolVersion": CONTRACT_VERSION,
            "jobId": job.job_id,
            "attempt": attempt,
            "attemptId": job.attempt_id,
            "operation": job.operation,
            "device": (
                {"backend": "cpu"}
                if device == "cpu"
                else {"backend": "cuda", "opaqueId": job.assigned_device_id}
            ),
            "inputs": [
                {
                    "schemaId": item.schema_id,
                    "ordinal": item.ordinal,
                    "commitRevision": item.commit_revision,
                    "dataContractSha256": item.data_contract_sha256,
                    "chunks": item.chunks,
                    "logicalRows": item.rows,
                    "nativeRows": list(item.native_rows),
                    "firstRangeOrdinal": item.first_range_ordinal,
                    "firstExampleOffset": item.first_example_offset,
                    "lastRangeOrdinal": item.last_range_ordinal,
                    "nextExampleOffset": item.next_example_offset,
                    "batches": item.batches,
                    "artifact": {
                        "path": item.absolute_path,
                        "byteCount": item.byte_count,
                        "sha256": item.sha256,
                    },
                }
                for item in inputs
            ],
            "inputRevision": job.input_revision,
            "inputClosed": job.input_state == InputState.CLOSED,
            "manifestSha256": job.manifest_sha256,
            "workspace": {"root": workspace},
            "model": model_manifest,
            "sourceEncoding": dict(job.source_encoding),
            "dataContract": dict(job.data_contract),
            "modelContract": dict(job.model_contract),
            "modelConfig": job.model_config.to_manifest(),
            "semanticDigests": dict(job.semantic_digests),
            "jobConfigSha256": job.config_hash,
        }
        if job.operation == "predict":
            model = self._validated_model(job)
            checkpoint = self.spool.model_absolute_path(model.checkpoint_path)
            document["predictionColumn"] = job.prediction_column
            model_manifest["checkpoint"] = {
                "path": checkpoint,
                "format": CHECKPOINT_FORMAT,
                "byteCount": model.byte_count,
                "checkpointSha256": model.sha256,
            }
        else:
            if job.model_label is None or job.training_config is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "fit job configuration is unavailable",
                )
            model_manifest["label"] = job.model_label
            initialization = validate_initialization(job.initialization)
            initialization_document = dict(initialization)
            if initialization["source"] == "publishedModel":
                model = self._validated_model(job)
                if model.sha256 != initialization["parentCheckpointSha256"]:
                    raise WorkerPlanError(
                        ErrorCode.MODEL_CORRUPT,
                        "parent checkpoint digest differs from fit initialization",
                    )
                if (
                    model.data_contract.get("dataContractSha256")
                    != initialization["parentDataContractSha256"]
                ):
                    raise WorkerPlanError(
                        ErrorCode.MODEL_CORRUPT,
                        "parent data contract digest differs from fit initialization",
                    )
                if (
                    job.data_contract.get("dataContractSha256")
                    != initialization["dataContractSha256"]
                ):
                    raise WorkerPlanError(
                        ErrorCode.INTERNAL,
                        "fit data contract digest differs from initialization",
                    )
                model_manifest["parentCheckpoint"] = {
                    "path": self.spool.model_absolute_path(
                        model.checkpoint_path
                    ),
                    "format": CHECKPOINT_FORMAT,
                    "byteCount": model.byte_count,
                    "checkpointSha256": model.sha256,
                }
            document["initialization"] = initialization_document
            document["training"] = train_config_to_manifest(
                job.training_config
            )
            document["diagnostics"] = (
                job.training_config.diagnostics.to_document()
            )
            document["recovery"] = None
            if self.recovery_store is not None:
                if any(item.storage_class != "recovery" for item in inputs):
                    raise WorkerPlanError(
                        ErrorCode.RECOVERY_INPUT_UNAVAILABLE,
                        "resumable fit requires persistent committed inputs",
                    )
                if (
                    job.input_state == InputState.CLOSED
                    and job.manifest_sha256 is None
                ):
                    raise WorkerPlanError(
                        ErrorCode.INTERNAL,
                        "closed fit manifest hash is unavailable",
                    )
                if (
                    job.resume_generation is not None
                    and recovery_checkpoint is None
                ):
                    raise WorkerPlanError(
                        ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                        "claimed recovery generation is unavailable",
                    )
                if recovery_checkpoint is not None:
                    path = self._validated_recovery_checkpoint(
                        job,
                        recovery_checkpoint,
                    )
                    recovery: JsonObject = {
                        "format": RECOVERY_FORMAT,
                        "jobId": job.job_id,
                        "generation": recovery_checkpoint.generation,
                        "inputRevision": recovery_checkpoint.input_revision,
                        "jobConfigSha256": job.config_hash,
                        "semanticDigests": dict(job.semantic_digests),
                        "manifestSha256": job.manifest_sha256,
                        "checkpoint": {
                            "path": path,
                            "format": CHECKPOINT_FORMAT,
                            "byteCount": recovery_checkpoint.byte_count,
                            "checkpointSha256": recovery_checkpoint.sha256,
                        },
                        "progress": {
                            "completedEpochs": recovery_checkpoint.completed_epochs,
                            "globalStep": recovery_checkpoint.global_step,
                            "trainingComplete": recovery_checkpoint.training_complete,
                        },
                    }
                    document["recovery"] = recovery
        validate_document(document, "command-manifest")
        try:
            self.spool.write_json_once(manifest_path, document)
        except FileExistsError as exc:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "worker command manifest already exists",
            ) from exc
        return manifest_path, workspace

    def _validated_model(
        self,
        job: ExecutionJobRecord,
    ) -> ModelArtifactRecord:
        model_ref = job.input_model_ref
        if model_ref is None:
            raise WorkerPlanError(
                ErrorCode.MODEL_SCHEMA_MISMATCH,
                "job does not identify a model generation",
            )
        model = self.ledger.get_model_artifact(
            model_ref,
            owner_subject=job.owner_subject,
        )
        if model is None:
            raise WorkerPlanError(
                ErrorCode.NOT_FOUND,
                "resolved model generation was not found",
            )
        try:
            checkpoint = self.spool.model_absolute_path(
                model.checkpoint_path,
            )
            expected = self.spool.model_checkpoint_path(model.model_ref)
        except ValueError as exc:
            raise WorkerPlanError(
                ErrorCode.MODEL_CORRUPT,
                "resolved model checkpoint identity is invalid",
            ) from exc
        if checkpoint != expected:
            raise WorkerPlanError(
                ErrorCode.MODEL_CORRUPT,
                "resolved model checkpoint identity is invalid",
            )
        try:
            byte_count = os.path.getsize(checkpoint)
            checkpoint_sha256 = _sha256_file(checkpoint)
        except OSError as exc:
            raise WorkerPlanError(
                ErrorCode.MODEL_UNAVAILABLE,
                "resolved model checkpoint is unavailable",
            ) from exc
        if byte_count != model.byte_count or checkpoint_sha256 != model.sha256:
            raise WorkerPlanError(
                ErrorCode.MODEL_CORRUPT,
                "resolved model checkpoint integrity validation failed",
            )
        return model

    def _validated_recovery_checkpoint(
        self,
        job: ExecutionJobRecord,
        checkpoint: TrainingRecoveryCheckpointRecord,
    ) -> str:
        if checkpoint.generation != job.resume_generation:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "claimed recovery generation does not match the ledger",
            )
        recovery_store = self.recovery_store
        if recovery_store is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "recovery storage is unavailable",
            )
        path = recovery_store.absolute_path(checkpoint.relative_path)
        expected = recovery_store.checkpoint_path(
            job.job_id,
            checkpoint.generation,
        )
        if (
            path != expected
            or not os.path.isfile(path)
            or os.path.getsize(path) != checkpoint.byte_count
        ):
            raise WorkerPlanError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "registered training recovery checkpoint is unavailable",
            )
        if _sha256_file(path) != checkpoint.sha256:
            raise WorkerPlanError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "registered training recovery checkpoint is corrupt",
            )
        return path

    def _validated_inputs(
        self,
        job: ExecutionJobRecord,
        *,
        start_ordinal: int = 0,
    ) -> tuple[ExecutionInput, ...]:
        inputs = tuple(
            item
            for item in self.ledger.list_committed_inputs(job.job_id)
            if start_ordinal <= item.ordinal < job.input_frame_count
        )
        if [item.ordinal for item in inputs] != list(
            range(start_ordinal, job.input_frame_count)
        ):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "contiguous input ordinals are inconsistent",
            )
        prepared: list[ExecutionInput] = []
        expected_schema_id = (
            FIT_INPUT_SCHEMA_ID
            if job.operation == "fit"
            else PREDICT_INPUT_SCHEMA_ID
        )
        for item in inputs:
            if item.schema_id != expected_schema_id:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "committed input schema differs from the job operation",
                )
            persistent = item.storage_class == "recovery"
            store = (
                self.recovery_store
                if persistent
                else self.spool
            )
            if store is None:
                raise WorkerPlanError(
                    ErrorCode.RECOVERY_INPUT_UNAVAILABLE,
                    "persistent fit input storage is unavailable",
                )
            actual = store.absolute_path(item.relative_path)
            if os.path.dirname(actual) != store.input_directory(job.job_id):
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "committed input path is invalid",
                )
            try:
                size = os.path.getsize(actual)
            except OSError as exc:
                raise WorkerPlanError(
                    (
                        ErrorCode.RECOVERY_INPUT_UNAVAILABLE
                        if persistent
                        else ErrorCode.INTERNAL
                    ),
                    "committed input is unavailable",
                ) from exc
            if size != item.byte_count or size > self.config.max_payload_bytes:
                raise WorkerPlanError(
                    (
                        ErrorCode.RECOVERY_INPUT_UNAVAILABLE
                        if persistent
                        else ErrorCode.INTERNAL
                    ),
                    "committed input size is invalid",
                )
            if _sha256_file(actual) != item.sha256:
                raise WorkerPlanError(
                    (
                        ErrorCode.RECOVERY_INPUT_UNAVAILABLE
                        if persistent
                        else ErrorCode.INTERNAL
                    ),
                    "committed input digest is invalid",
                )
            prepared.append(ExecutionInput(
                ordinal=item.ordinal,
                commit_revision=item.commit_revision,
                schema_id=item.schema_id,
                data_contract_sha256=item.data_contract_sha256,
                chunks=item.chunks,
                rows=item.rows,
                native_rows=item.native_rows,
                first_range_ordinal=item.first_range_ordinal,
                first_example_offset=item.first_example_offset,
                last_range_ordinal=item.last_range_ordinal,
                next_example_offset=item.next_example_offset,
                batches=item.batches,
                byte_count=item.byte_count,
                sha256=item.sha256,
                absolute_path=actual,
                storage_class=item.storage_class,
            ))
        return tuple(prepared)

    def streaming_inputs(
        self,
        job: ExecutionJobRecord,
        start_ordinal: int,
    ) -> tuple[ExecutionInput, ...]:
        """Validate the newly visible contiguous suffix for one attempt."""

        if start_ordinal < 0 or start_ordinal > job.input_frame_count:
            raise ValueError("invalid streaming input ordinal")
        return self._validated_inputs(job, start_ordinal=start_ordinal)

def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
