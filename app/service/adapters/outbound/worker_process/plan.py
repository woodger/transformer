from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence

from app.contracts.worker.v1 import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    validate_document,
)
from app.contracts.worker.v1.config import (
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.service.application.ports.workers import ExecutionInput, ExecutionPlan
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.job import ErrorCode
from app.service.domain.records import ExecutionJobRecord

_COPY_CHUNK_BYTES = 1024 * 1024
_FIT_SPOOL_OPTION = "--input-spool-dir"


class WorkerPlanError(AttemptExecutionError):
    pass


class WorkerPlanBuilder:
    """Validate durable artifacts and render one trusted CLI execution plan."""

    def __init__(
        self,
        config,
        ledger,
        spool,
        recovery_store=None,
        *,
        python_executable: str,
        cli_path: str,
    ):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.recovery_store = recovery_store
        self.python_executable = python_executable
        self.cli_path = cli_path

    def build(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        *,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
    ) -> ExecutionPlan:
        inputs = self._validated_inputs(job)
        recovery_checkpoint = (
            self.ledger.latest_recovery_checkpoint(job.job_id)
            if (
                job.operation == "fit"
                and self.recovery_store is not None
            )
            else None
        )
        if argv_hook is not None:
            argv = self.build_argv(job, attempt, argv_hook=argv_hook)
            return ExecutionPlan(
                inputs=inputs,
                argv=argv,
                uses_spooled_fit=(
                    job.operation == "fit" and _FIT_SPOOL_OPTION in argv
                ),
                uses_training_recovery=(
                    job.operation == "fit" and self.recovery_store is not None
                ),
                resume_training_complete=bool(
                    recovery_checkpoint is not None
                    and recovery_checkpoint.training_complete
                ),
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
            uses_spooled_fit=False,
            uses_training_recovery=(
                job.operation == "fit"
                and self.recovery_store is not None
            ),
            resume_training_complete=bool(
                recovery_checkpoint is not None
                and recovery_checkpoint.training_complete
            ),
            protocol_version=CONTRACT_VERSION,
            manifest_path=manifest_path,
            workspace=workspace,
        )

    def build_argv(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        *,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
        job_mapping: dict | None = None,
    ) -> tuple[str, ...]:
        if not isinstance(attempt, int) or attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        if argv_hook is None:
            if job.attempt_id is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "claimed job has no attempt identity",
                )
            return self._worker_argv(
                job,
                attempt,
                self.spool.attempt_manifest_path(job.job_id, attempt),
            )
        argv = self._legacy_argv(job, attempt)
        if argv_hook is not None:
            if job_mapping is None:
                job_mapping = self.ledger.get_job(job.job_id)
            if job_mapping is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "claimed job disappeared",
                )
            argv = list(argv_hook(job_mapping, tuple(argv)))
        if not argv or any(
            not isinstance(value, str) or "\x00" in value
            for value in argv
        ):
            raise ValueError("worker argv hook returned invalid argv")
        return tuple(argv)

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
        recovery_checkpoint,
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
                "CUDA attempt has no assigned physical device",
            )
        model_config = job.model_config
        if model_config is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "worker model configuration is unavailable",
            )
        document = {
            "contract": CONTRACT_NAME,
            "protocolVersion": CONTRACT_VERSION,
            "jobId": job.job_id,
            "attempt": attempt,
            "attemptId": job.attempt_id,
            "operation": job.operation,
            "device": (
                {"kind": "cpu"}
                if device == "cpu"
                else {"kind": "cuda", "opaqueId": job.assigned_device_id}
            ),
            "inputs": [
                {
                    "schemaId": item.schema_id,
                    "ordinal": item.ordinal,
                    "rows": item.rows,
                    "artifact": {
                        "path": item.absolute_path,
                        "byteCount": item.byte_count,
                        "sha256": item.sha256,
                    },
                }
                for item in inputs
            ],
            "workspace": {"root": workspace},
            "model": {"config": model_config_to_manifest(model_config)},
        }
        if job.operation == "predict":
            model = self._validated_model(job)
            checkpoint = self.spool.model_absolute_path(model.checkpoint_path)
            document["predictionColumn"] = job.prediction_column
            document["model"]["checkpoint"] = {
                "path": checkpoint,
                "byteCount": os.path.getsize(checkpoint),
                "sha256": model.sha256,
            }
        else:
            if job.model_label is None or job.training_config is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "fit job configuration is unavailable",
                )
            document["model"]["label"] = job.model_label
            document["training"] = train_config_to_manifest(
                job.training_config
            )
            if self.recovery_store is not None:
                if any(item.storage_class != "recovery" for item in inputs):
                    raise WorkerPlanError(
                        ErrorCode.RECOVERY_INPUT_UNAVAILABLE,
                        "resumable fit requires persistent committed inputs",
                    )
                if job.seal_hash is None:
                    raise WorkerPlanError(
                        ErrorCode.INTERNAL,
                        "sealed fit manifest hash is unavailable",
                    )
                recovery = {
                    "configSha256": job.config_hash,
                    "manifestSha256": job.seal_hash,
                }
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
                    recovery["checkpoint"] = {
                        "path": path,
                        "byteCount": recovery_checkpoint.byte_count,
                        "sha256": recovery_checkpoint.sha256,
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

    def _validated_model(self, job: ExecutionJobRecord):
        model = self.ledger.get_model_artifact(
            job.input_model_ref,
            owner_subject=job.owner_subject,
        )
        if model is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "resolved model generation is unavailable",
            )
        checkpoint = self.spool.model_absolute_path(model.checkpoint_path)
        expected = self.spool.model_checkpoint_path(model.model_ref)
        if checkpoint != expected or not os.path.isfile(checkpoint):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "resolved model checkpoint is unavailable",
            )
        if _sha256_file(checkpoint) != model.sha256:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "resolved model checkpoint digest is invalid",
            )
        return model

    def _validated_recovery_checkpoint(self, job, checkpoint) -> str:
        if checkpoint.generation != job.resume_generation:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "claimed recovery generation does not match the ledger",
            )
        path = self.recovery_store.absolute_path(checkpoint.relative_path)
        expected = self.recovery_store.checkpoint_path(
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
    ) -> tuple[ExecutionInput, ...]:
        inputs = self.ledger.list_committed_inputs(job.job_id)
        if [item.ordinal for item in inputs] != list(range(len(inputs))):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "sealed input ordinals are inconsistent",
            )
        prepared = []
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
            expected = store.input_path(job.job_id, item.ordinal)
            actual = store.absolute_path(item.relative_path)
            if actual != expected:
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
                schema_id=item.schema_id,
                rows=item.rows,
                byte_count=item.byte_count,
                sha256=item.sha256,
                absolute_path=actual,
                storage_class=item.storage_class,
            ))
        return tuple(prepared)

    def _legacy_argv(
        self,
        job: ExecutionJobRecord,
        attempt: int,
    ) -> list[str]:
        device = job.selected_device
        if device not in ("cpu", "cuda"):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "queued job has no selected device",
            )
        base = [self.python_executable, self.cli_path]
        if job.operation == "predict":
            model = self.ledger.get_model_artifact(
                job.input_model_ref,
                owner_subject=job.owner_subject,
            )
            if model is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "resolved model generation is unavailable",
                )
            checkpoint = self.spool.model_absolute_path(
                model.checkpoint_path
            )
            expected = self.spool.model_checkpoint_path(model.model_ref)
            if checkpoint != expected or not os.path.isfile(checkpoint):
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "resolved model checkpoint is unavailable",
                )
            if _sha256_file(checkpoint) != model.sha256:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "resolved model checkpoint digest is invalid",
                )
            return [
                *base,
                "predict-stream",
                "--device", device,
                "--checkpoint", checkpoint,
                "--pred-col", job.prediction_column,
                "--max-frame-bytes", str(self.config.max_payload_bytes),
            ]

        model = job.model_config
        train = job.training_config
        if model is None or train is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "fit job configuration is unavailable",
            )
        input_directory = (
            self.recovery_store.input_directory(job.job_id)
            if self.recovery_store is not None
            else self.spool.input_directory(job.job_id)
        )
        argv = [
            *base,
            "fit-stream",
            "--device", device,
            "--checkpoint-out", self.spool.attempt_checkpoint_path(
                job.job_id,
                attempt,
            ),
            "--metrics-out", self.spool.attempt_metrics_path(
                job.job_id,
                attempt,
            ),
            "--max-frame-bytes", str(self.config.max_payload_bytes),
            _FIT_SPOOL_OPTION, input_directory,
            "--input-frame-count", str(job.input_frame_count),
            "--seq-len", str(model.seq_len),
            "--hidden", str(model.hidden),
            "--layers", str(model.layers),
            "--dropout", str(model.dropout),
            "--nhead", str(model.nhead),
            "--mode", model.context_mode,
            "--lr", str(train.lr),
            "--weight-decay", str(train.weight_decay),
            "--batch-size", str(train.batch_size),
            "--epochs", str(train.epochs),
            "--loss-stage", str(train.loss_stage),
            "--loss-schedule", train.loss_schedule,
            "--stage-size", str(train.stage_size),
            "--patience", str(train.patience),
            "--monitor", train.monitor,
            "--monitor-min-improvement", str(
                train.monitor_min_improvement
            ),
            "--seed", str(train.seed),
            (
                "--save-best-checkpoint"
                if train.save_best_checkpoint
                else "--no-save-best-checkpoint"
            ),
        ]
        if train.use_amp:
            argv.append("--use-amp")
        if train.deterministic:
            argv.append("--deterministic")
        if self.recovery_store is not None:
            if job.seal_hash is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "sealed fit manifest hash is unavailable",
                )
            if any(
                item.storage_class != "recovery"
                for item in self.ledger.list_committed_inputs(
                    job.job_id
                )
            ):
                raise WorkerPlanError(
                    ErrorCode.RECOVERY_INPUT_UNAVAILABLE,
                    "fit inputs are not stored persistently",
                )
            argv.extend([
                "--recovery-checkpoint-dir",
                self.recovery_store.checkpoint_directory(job.job_id),
                "--recovery-events-out",
                self.spool.attempt_recovery_events_path(
                    job.job_id,
                    attempt,
                ),
                "--recovery-config-hash",
                job.config_hash,
                "--recovery-seal-hash",
                job.seal_hash,
            ])
            checkpoint = self.ledger.latest_recovery_checkpoint(
                job.job_id
            )
            checkpoint_generation = (
                None
                if checkpoint is None
                else checkpoint.generation
            )
            if checkpoint_generation != job.resume_generation:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "claimed recovery generation does not match the ledger",
                )
            if checkpoint is not None:
                path = self.recovery_store.absolute_path(
                    checkpoint.relative_path
                )
                expected = self.recovery_store.checkpoint_path(
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
                argv.extend(["--resume-checkpoint", path])
        return argv


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
