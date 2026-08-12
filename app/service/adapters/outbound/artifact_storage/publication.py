from __future__ import annotations

import hashlib
import math
import os
import shutil
import uuid

from app.contracts.flight.v4.arrow import validate_prediction_file
from app.contracts.worker.v3 import PREDICTION_OUTPUT_SCHEMA_ID
from app.contracts.worker.v3.config import (
    ModelConfig,
    TrainConfig,
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.contracts.worker.v3.objective import (
    CHECKPOINT_FORMAT,
    ml_contract,
    objective_config,
    objective_config_sha256,
)
from app.service.application.ports.workers import ExecutionInput
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, ExecutionState
from app.service.domain.records import ExecutionJobRecord, StagedPredictionOutput

_COPY_CHUNK_BYTES = 1024 * 1024


class WorkerArtifactError(AttemptExecutionError):
    pass


class WorkerArtifactPublisher:
    """Stage, validate, publish, and clean up worker-owned artifacts."""

    def __init__(
        self,
        ledger,
        spool,
        *,
        logger,
        metrics,
        max_payload_bytes: int | None = None,
    ):
        self.ledger = ledger
        self.spool = spool
        self.logger = logger
        self.metrics = metrics
        self.max_payload_bytes = max_payload_bytes

    def _publish_outputs(
        self,
        job: ExecutionJobRecord,
        inputs: tuple[ExecutionInput, ...],
        outputs: tuple[StagedPredictionOutput, ...],
    ) -> None:
        if len(outputs) != len(inputs):
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess output count did not match input count",
            )
        records = [item.ledger_record() for item in outputs]
        result = {
            "outputs": [
                {"ordinal": item.ordinal, "rows": item.rows}
                for item in outputs
            ]
        }
        self.ledger.publish_outputs(
            job.job_id,
            job.attempt,
            records,
            attempt_id=job.attempt_id,
            result=result,
        )
        for item in outputs:
            self.metrics.add("predictionOutputBytes", item.byte_count)
            self.metrics.add("predictionOutputRows", item.rows)
            self.metrics.add("predictionOutputBatches", item.batches)
            self.logger.event(
                "flight.output.published",
                jobId=job.job_id,
                ordinal=item.ordinal,
                rows=item.rows,
                batches=item.batches,
                bytes=item.byte_count,
            )

    def publish_outputs_from_manifest(
        self,
        job: ExecutionJobRecord,
        inputs: tuple[ExecutionInput, ...],
        result: dict,
    ) -> None:
        raw_outputs = result.get("artifacts")
        if not isinstance(raw_outputs, list) or len(raw_outputs) != len(inputs):
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction worker output count did not match input count",
            )
        outputs = []
        for item, document in zip(inputs, raw_outputs, strict=True):
            if (
                document.get("schemaId") != PREDICTION_OUTPUT_SCHEMA_ID
                or document.get("ordinal") != item.ordinal
                or document.get("rows") != item.rows
            ):
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction worker output manifest differs from the inputs",
                )
            artifact = document["artifact"]
            path = os.path.abspath(os.fspath(artifact["path"]))
            expected = self.spool.attempt_output_path(
                job.job_id,
                job.attempt,
                item.ordinal,
            )
            try:
                byte_count = os.path.getsize(path)
            except OSError as exc:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction worker output is unavailable",
                ) from exc
            if (
                path != expected
                or byte_count != artifact["byteCount"]
                or (
                    self.max_payload_bytes is not None
                    and byte_count > self.max_payload_bytes
                )
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction worker output integrity check failed",
                )
            try:
                stats = validate_prediction_file(
                    path,
                    job.prediction_column,
                    item.rows,
                )
            except (OSError, ServiceError, ValueError) as exc:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction worker output is not valid Arrow",
                ) from exc
            outputs.append(StagedPredictionOutput(
                ordinal=item.ordinal,
                rows=stats.rows,
                batches=stats.batches,
                byte_count=byte_count,
                sha256=artifact["sha256"],
                schema_fingerprint=stats.schema_fingerprint,
                relative_path=self.spool.relative_path(path),
            ))
        self._publish_outputs(job, inputs, tuple(outputs))

    def publish_model_from_manifest(
        self,
        job: ExecutionJobRecord,
        result: dict,
    ) -> None:
        checkpoint = result.get("checkpoint")
        metrics = result.get("metrics")
        if not isinstance(checkpoint, dict) or not isinstance(metrics, dict):
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit worker result does not contain required artifacts",
            )
        expected = (
            (
                checkpoint,
                self.spool.attempt_checkpoint_path(job.job_id, job.attempt),
            ),
            (
                metrics,
                self.spool.attempt_metrics_path(job.job_id, job.attempt),
            ),
        )
        for artifact, expected_path in expected:
            path = os.path.abspath(os.fspath(artifact["path"]))
            try:
                valid = (
                    path == expected_path
                    and os.path.getsize(path) == artifact["byteCount"]
                    and _sha256_file(path) == artifact["sha256"]
                )
            except OSError:
                valid = False
            if not valid:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit worker artifact integrity check failed",
                )
        self._publish_model(job, result["checkpointMetadata"])

    def _publish_model(
        self,
        job: ExecutionJobRecord,
        checkpoint_metadata: dict,
    ) -> None:
        attempt_path = self.spool.attempt_checkpoint_path(
            job.job_id, job.attempt
        )
        if not os.path.isfile(attempt_path):
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess did not create a checkpoint",
            )
        try:
            actual_model = ModelConfig.from_dict(
                checkpoint_metadata.get("modelConfig")
            )
            actual_train = TrainConfig.from_dict(
                checkpoint_metadata.get("trainingConfig")
            )
        except Exception as exc:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        expected_model = job.model_config
        expected_train = job.training_config
        if expected_model is None or expected_train is None:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit job configuration is unavailable",
            )
        checkpoint_format = checkpoint_metadata.get("format")
        if checkpoint_format != CHECKPOINT_FORMAT or actual_model is None:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an unsupported checkpoint",
            )
        expected_values = expected_model.to_dict()
        expected_values["feature_dim"] = job.feature_dim
        if actual_model.to_dict() != expected_values or actual_train != expected_train:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit checkpoint configuration differs from the immutable job config",
            )

        digest = _sha256_file(attempt_path)
        byte_count = os.path.getsize(attempt_path)
        model_ref = f"mdl_{uuid.uuid4().hex}"
        model_directory = self.spool.model_directory(model_ref)
        checkpoint_path = self.spool.model_checkpoint_path(model_ref)
        metadata_path = self.spool.model_metadata_path(model_ref)
        try:
            service_version = checkpoint_metadata.get("serviceVersion")
            if checkpoint_metadata.get("dataContract") != (
                _data_contract_to_api(job.data_contract)
            ):
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit checkpoint data contract differs from the job",
                )
            expected_ml_contract = ml_contract(actual_train)
            if checkpoint_metadata.get("mlContract") != expected_ml_contract:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit checkpoint ML contract differs from the job",
                )
            expected_objective = objective_config(actual_train)
            if checkpoint_metadata.get("objectiveConfig") != expected_objective:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit checkpoint objective configuration differs from the job",
                )
            data_schema = _canonical_data_schema(actual_model)
            checkpoint_selection = _validate_worker_selection(
                checkpoint_metadata.get("checkpointSelection"),
                actual_train,
            )
            safe_checkpoint = {
                "format": checkpoint_format,
                "serviceVersion": service_version,
                "sha256": digest,
                "bytes": byte_count,
                "modelConfig": model_config_to_manifest(actual_model),
                "trainingConfig": train_config_to_manifest(actual_train),
                "dataContract": _data_contract_to_api(job.data_contract),
                "mlContract": expected_ml_contract,
                "checkpointSelection": checkpoint_selection,
            }
        except WorkerArtifactError:
            raise
        except Exception as exc:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        if not isinstance(safe_checkpoint["serviceVersion"], str) or not safe_checkpoint[
            "serviceVersion"
        ]:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit checkpoint does not contain a service version",
            )
        metadata = {
            "modelRef": model_ref,
            "label": job.model_label,
            # Internal snake_case copies let predict create resolve a generation
            # without depending on the public status document representation.
            "model_config": actual_model.to_dict(),
            "train_config": actual_train.to_dict(),
            "data_contract": dict(job.data_contract),
            "ml_contract": expected_ml_contract,
            "objective_config": expected_objective,
            "data_schema": data_schema,
            "checkpoint": safe_checkpoint,
        }
        try:
            with open(attempt_path, "rb") as source:
                with self.spool.staged_file(checkpoint_path) as (target, _):
                    shutil.copyfileobj(source, target, _COPY_CHUNK_BYTES)
            self.spool.atomic_write_json(metadata_path, metadata)
            self.ledger.publish_model(
                job.job_id,
                job.attempt,
                attempt_id=job.attempt_id,
                model_ref=model_ref,
                label=job.model_label,
                generation=None,
                checkpoint_path=self.spool.model_relative_path(checkpoint_path),
                metadata_path=self.spool.model_relative_path(metadata_path),
                byte_count=byte_count,
                sha256=digest,
                metadata=metadata,
                result={"modelRef": model_ref, "checkpoint": safe_checkpoint},
            )
            self.metrics.add("checkpointBytes", byte_count)
            self.logger.event(
                "flight.model.published",
                jobId=job.job_id,
                modelRef=model_ref,
                bytes=byte_count,
                sha256=digest,
            )
        except BaseException:
            current = self.ledger.get_execution_job(job.job_id)
            if (
                current is None
                or current.execution_state != ExecutionState.SUCCEEDED
            ):
                self.spool.remove(model_directory)
            raise

    def cleanup_unpublished(self, job: ExecutionJobRecord) -> None:
        try:
            inputs = self.ledger.list_committed_inputs(job.job_id)
        except Exception as exc:
            self.logger.event(
                "flight.worker.cleanup_failed",
                jobId=job.job_id,
                attempt=job.attempt,
                artifact="attempt",
                errorType=type(exc).__name__,
            )
            return
        paths = [
            self.spool.attempt_output_path(
                job.job_id, job.attempt, item.ordinal
            )
            for item in inputs
        ]
        paths.append(
            self.spool.attempt_checkpoint_path(job.job_id, job.attempt)
        )
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                self.spool.remove(path)
            except Exception as exc:
                # Publication is ledger-gated. A filesystem cleanup failure
                # leaves only an invisible orphan for startup reconciliation.
                self.logger.event(
                    "flight.worker.cleanup_failed",
                    jobId=job.job_id,
                    attempt=job.attempt,
                    artifact="attempt",
                    errorType=type(exc).__name__,
                )

def _data_contract_to_api(value: dict) -> dict:
    return {
        "id": value["id"],
        "version": value["version"],
        "dataContractSha256": value["data_contract_sha256"],
        "seqLen": value["seq_len"],
        "featureDim": value["feature_dim"],
        "targetSchemaId": value["target_schema_id"],
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_selection_to_api(
    value: dict | None,
    train_config: TrainConfig,
) -> dict:
    if not isinstance(value, dict):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        )
    enabled = train_config.selection is not None
    score = value.get("bestSelectionScore")
    frame = value.get("bestFrame")
    epoch = value.get("bestEpoch")
    valid_frame = frame is None or (type(frame) is int and frame >= 0)
    valid_epoch = epoch is None or (type(epoch) is int and epoch >= 1)
    if enabled:
        valid_selection = (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and epoch is not None
            and value.get("source") == "best_selection_score"
        )
    else:
        valid_selection = (
            score is None
            and frame is None
            and epoch is None
            and value.get("source") == "last_maximum_stage"
        )
    if (
        set(value)
        != {
            "enabled",
            "objectiveConfigSha256",
            "bestSelectionScore",
            "bestFrame",
            "bestEpoch",
            "source",
        }
        or value.get("enabled") is not enabled
        or value.get("objectiveConfigSha256")
        != objective_config_sha256(train_config)
        or not valid_frame
        or not valid_epoch
        or not valid_selection
    ):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        )
    return dict(value)


def _canonical_data_schema(model_config: ModelConfig) -> dict:
    feature_dim = model_config.feature_dim
    if feature_dim is None:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit worker metadata has no feature dimension",
        )
    return {
        "schemaVersion": 2,
        "tensorDtype": "float32",
        "source": {
            "column": "src",
            "acceptedElementTypes": ["float32"],
            "width": model_config.seq_len * feature_dim,
        },
        "target": {
            "column": "tgt",
            "acceptedElementTypes": ["float32"],
            "width": 6,
            "targetSchemaId": "inventory.target.v1",
        },
        "featureDim": feature_dim,
        "modelInputFeatureDim": (
            feature_dim * 2
            if model_config.context_mode == "relaxed"
            else feature_dim
        ),
        "contextMode": model_config.context_mode,
        "normalization": None,
        "missing": {
            "nanFill": 0.0,
            "flags": (
                "per-feature"
                if model_config.context_mode == "relaxed"
                else "none"
            ),
        },
    }


def _validate_worker_selection(
    value: dict | None,
    train_config: TrainConfig,
) -> dict:
    if not isinstance(value, dict):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit worker checkpoint selection metadata is invalid",
        )
    result = _checkpoint_selection_to_api(value, train_config)
    if result != value:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit worker checkpoint selection metadata is invalid",
        )
    return result
