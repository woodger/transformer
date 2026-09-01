from __future__ import annotations

import hashlib
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import BinaryIO, Protocol, cast

from app.contracts.flight.v8.arrow import validate_prediction_file
from app.contracts.json_types import JsonObject
from app.contracts.worker.v9 import PREDICTION_OUTPUT_SCHEMA_ID
from app.contracts.worker.v9.config import (
    ModelConfig,
    TrainConfig,
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.contracts.worker.v9.diagnostics import DiagnosticsConfig
from app.contracts.worker.v9.objective import (
    CHECKPOINT_FORMAT,
    ObjectiveConfig,
    objective_config_sha256,
    objective_from_ml_contract,
)
from app.service.application.ports.artifacts import PublishedModelArtifacts
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.workers import ExecutionInput
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.errors import ServiceError
from app.service.domain.initialization import validate_initialization
from app.service.domain.job import ErrorCode, ExecutionState
from app.service.domain.records import (
    CommittedInputRecord,
    ExecutionJobRecord,
    StagedPredictionOutput,
)

_COPY_CHUNK_BYTES = 1024 * 1024


class _PublicationLedger(Protocol):
    def publish_outputs(
        self,
        job_id: str,
        attempt: int,
        outputs: Sequence[JsonObject],
        *,
        attempt_id: str,
        result: JsonObject,
    ) -> Mapping[str, object]: ...

    def publish_model(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        model_ref: str,
        label: str,
        generation: int | None,
        checkpoint_path: str,
        metadata_path: str,
        byte_count: int,
        sha256: str,
        metadata: JsonObject,
        result: JsonObject,
        now: float | None = None,
    ) -> Mapping[str, object]: ...

    def get_execution_job(self, job_id: str) -> ExecutionJobRecord | None: ...

    def list_committed_inputs(
        self,
        job_id: str,
    ) -> Sequence[CommittedInputRecord]: ...


class _PublicationSpool(Protocol):
    def attempt_output_path(
        self,
        job_id: str,
        attempt: int,
        ordinal: int,
    ) -> str: ...

    def attempt_checkpoint_path(self, job_id: str, attempt: int) -> str: ...

    def relative_path(self, absolute_path: str) -> str: ...

    def model_directory(self, model_ref: str) -> str: ...

    def model_checkpoint_path(self, model_ref: str) -> str: ...

    def model_metadata_path(self, model_ref: str) -> str: ...

    def model_relative_path(self, absolute_path: str) -> str: ...

    def staged_file(
        self,
        destination: str,
    ) -> AbstractContextManager[tuple[BinaryIO, str]]: ...

    def atomic_write_json(
        self,
        destination: str,
        document: JsonObject,
    ) -> str: ...

    def remove(self, path: str) -> bool: ...


class WorkerArtifactError(AttemptExecutionError):
    pass


class WorkerArtifactPublisher:
    """Stage, validate, publish, and clean up worker-owned artifacts."""

    def __init__(
        self,
        ledger: _PublicationLedger,
        spool: _PublicationSpool,
        *,
        logger: EventLogger,
        metrics: OperationalMetricSink,
        max_payload_bytes: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ledger = ledger
        self.spool = spool
        self.logger = logger
        self.metrics = metrics
        self.max_payload_bytes = max_payload_bytes
        self._monotonic = monotonic

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
        result: JsonObject = {
            "outputs": [
                {"ordinal": item.ordinal, "rows": item.rows}
                for item in outputs
            ]
        }
        self.ledger.publish_outputs(
            job.job_id,
            job.attempt,
            records,
            attempt_id=_attempt_id(job),
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
        result: JsonObject,
    ) -> None:
        raw_outputs = result.get("artifacts")
        if not isinstance(raw_outputs, list) or len(raw_outputs) != len(inputs):
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction worker output count did not match input count",
            )
        outputs: list[StagedPredictionOutput] = []
        for item, raw_document in zip(inputs, raw_outputs, strict=True):
            document = _object(raw_document, "prediction output manifest")
            if (
                document.get("schemaId") != PREDICTION_OUTPUT_SCHEMA_ID
                or document.get("ordinal") != item.ordinal
                or document.get("rows") != item.rows
            ):
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "prediction worker output manifest differs from the inputs",
                )
            artifact = _object(
                document.get("artifact"),
                "prediction output artifact",
            )
            path = os.path.abspath(
                os.fspath(_string(artifact.get("path"), "artifact path"))
            )
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
                or byte_count
                != _integer(artifact.get("byteCount"), "artifact byteCount")
                or (
                    self.max_payload_bytes is not None
                    and byte_count > self.max_payload_bytes
                )
                or _sha256_file(path)
                != _string(artifact.get("sha256"), "artifact sha256")
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
                    objective_from_ml_contract(job.ml_contract).targets,
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
                sha256=_string(artifact.get("sha256"), "artifact sha256"),
                schema_fingerprint=stats.schema_fingerprint,
                relative_path=self.spool.relative_path(path),
            ))
        self._publish_outputs(job, inputs, tuple(outputs))

    def publish_model_from_manifest(
        self,
        job: ExecutionJobRecord,
        result: JsonObject,
    ) -> PublishedModelArtifacts:
        try:
            checkpoint = _object(
                result.get("checkpoint"),
                "fit checkpoint artifact",
            )
        except ValueError as exc:
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit worker result does not contain required artifacts",
            ) from exc
        expected = ((
            checkpoint,
            self.spool.attempt_checkpoint_path(job.job_id, job.attempt),
        ),)
        for artifact, expected_path in expected:
            path = os.path.abspath(os.fspath(
                _string(artifact.get("path"), "fit artifact path")
            ))
            try:
                valid = (
                    path == expected_path
                    and os.path.getsize(path)
                    == _integer(
                        artifact.get("byteCount"),
                        "fit artifact byteCount",
                    )
                    and _sha256_file(path)
                    == _string(
                        artifact.get("sha256"),
                        "fit artifact sha256",
                    )
                )
            except OSError:
                valid = False
            if not valid:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit worker artifact integrity check failed",
                )
        try:
            checkpoint_metadata = _object(
                result.get("checkpointMetadata"),
                "fit checkpoint metadata",
            )
        except ValueError as exc:
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit worker result contains invalid checkpoint metadata",
            ) from exc
        return self._publish_model(
            job,
            checkpoint_metadata,
        )

    def _publish_model(
        self,
        job: ExecutionJobRecord,
        checkpoint_metadata: JsonObject,
    ) -> PublishedModelArtifacts:
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
            if actual_train is not None:
                actual_train = replace(
                    actual_train,
                    diagnostics=DiagnosticsConfig.from_document(
                        checkpoint_metadata.get("diagnostics")
                    ),
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
        if (
            checkpoint_format != CHECKPOINT_FORMAT
            or actual_model is None
            or actual_train is None
        ):
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

        checkpoint_publication_started = self._monotonic()
        digest = _sha256_file(attempt_path)
        byte_count = os.path.getsize(attempt_path)
        terminal_checkpoint_publication_ms = (
            self._monotonic() - checkpoint_publication_started
        ) * 1000.0
        model_ref = f"mdl_{uuid.uuid4().hex}"
        model_directory = self.spool.model_directory(model_ref)
        checkpoint_path = self.spool.model_checkpoint_path(model_ref)
        metadata_path = self.spool.model_metadata_path(model_ref)
        try:
            service_version = _string(
                checkpoint_metadata.get("serviceVersion"),
                "fit checkpoint serviceVersion",
            )
            if checkpoint_metadata.get("dataContract") != (
                _data_contract_to_api(job.data_contract)
            ):
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit checkpoint data contract differs from the job",
                )
            expected_ml_contract = dict(job.ml_contract)
            if checkpoint_metadata.get("mlContract") != expected_ml_contract:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit checkpoint ML contract differs from the job",
                )
            objective = objective_from_ml_contract(expected_ml_contract)
            data_schema = _canonical_data_schema(actual_model)
            checkpoint_selection = _validate_worker_selection(
                checkpoint_metadata.get("checkpointSelection"),
                actual_train,
                objective,
            )
            initialization = validate_initialization(job.initialization)
            if checkpoint_metadata.get("initialization") != initialization:
                raise WorkerArtifactError(
                    ErrorCode.MALFORMED_OUTPUT,
                    "fit checkpoint initialization differs from the job",
                )
            safe_checkpoint: JsonObject = {
                "format": checkpoint_format,
                "serviceVersion": service_version,
                "sha256": digest,
                "bytes": byte_count,
                "modelConfig": model_config_to_manifest(actual_model),
                "trainingConfig": train_config_to_manifest(actual_train),
                "diagnostics": actual_train.diagnostics.to_document(),
                "dataContract": _data_contract_to_api(job.data_contract),
                "mlContract": expected_ml_contract,
                "objective": objective.to_document(),
                "initialization": initialization,
                "checkpointSelection": checkpoint_selection,
            }
        except WorkerArtifactError:
            raise
        except Exception as exc:
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        try:
            checkpoint_publication_started = self._monotonic()
            with open(attempt_path, "rb") as source:
                with self.spool.staged_file(checkpoint_path) as (target, _):
                    shutil.copyfileobj(source, target, _COPY_CHUNK_BYTES)
            terminal_checkpoint_publication_ms += (
                self._monotonic() - checkpoint_publication_started
            ) * 1000.0
            metadata: JsonObject = {
                "modelRef": model_ref,
                "label": job.model_label,
                # Internal snake_case copies let predict create resolve a generation
                # without depending on the public status document representation.
                "model_config": actual_model.to_dict(),
                "train_config": actual_train.to_dict(),
                "data_contract": dict(job.data_contract),
                "ml_contract": expected_ml_contract,
                "objective": objective.to_document(),
                "initialization": initialization,
                "data_schema": data_schema,
                "checkpoint": safe_checkpoint,
            }
            self.spool.atomic_write_json(metadata_path, metadata)
            self.ledger.publish_model(
                job.job_id,
                job.attempt,
                attempt_id=_attempt_id(job),
                model_ref=model_ref,
                label=_model_label(job),
                generation=None,
                checkpoint_path=self.spool.model_relative_path(checkpoint_path),
                metadata_path=self.spool.model_relative_path(metadata_path),
                byte_count=byte_count,
                sha256=digest,
                metadata=metadata,
                result={
                    "modelRef": model_ref,
                    "checkpoint": safe_checkpoint,
                },
            )
            self.metrics.add("checkpointBytes", byte_count)
            self.logger.event(
                "flight.model.published",
                jobId=job.job_id,
                modelRef=model_ref,
                bytes=byte_count,
                sha256=digest,
                initialization=initialization["kind"],
                parentModelRef=initialization.get("parentModelRef"),
            )
            return PublishedModelArtifacts(
                model_ref=model_ref,
                ml_contract=expected_ml_contract,
                checkpoint_publication_ms=(
                    terminal_checkpoint_publication_ms
                ),
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

def _data_contract_to_api(value: JsonObject) -> JsonObject:
    return {
        "id": value["id"],
        "version": value["version"],
        "profile": value["profile"],
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
    value: object,
    train_config: TrainConfig,
    objective: ObjectiveConfig,
) -> JsonObject:
    try:
        document = _object(value, "checkpoint selection")
    except ValueError as exc:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        ) from exc
    enabled = train_config.selection is not None
    score = document.get("bestSelectionScore")
    frame = document.get("bestFrame")
    epoch = document.get("bestEpoch")
    valid_frame = frame is None or (type(frame) is int and frame >= 0)
    valid_epoch = epoch is None or (type(epoch) is int and epoch >= 1)
    if enabled:
        valid_selection = (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and epoch is not None
            and document.get("source") == "best_selection_score"
        )
    else:
        valid_selection = (
            score is None
            and frame is None
            and epoch is None
            and document.get("source") == "last_epoch"
        )
    if (
        set(document)
        != {
            "enabled",
            "objectiveConfigSha256",
            "bestSelectionScore",
            "bestFrame",
            "bestEpoch",
            "source",
        }
        or document.get("enabled") is not enabled
        or document.get("objectiveConfigSha256")
        != objective_config_sha256(objective)
        or not valid_frame
        or not valid_epoch
        or not valid_selection
    ):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        )
    return dict(document)


def _canonical_data_schema(model_config: ModelConfig) -> JsonObject:
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
            "width": model_config.out_dim,
            "targetSchemaId": "inventory.target.v2",
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
    value: object,
    train_config: TrainConfig,
    objective: ObjectiveConfig,
) -> JsonObject:
    try:
        document = _object(value, "worker checkpoint selection")
    except ValueError as exc:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit worker checkpoint selection metadata is invalid",
        ) from exc
    result = _checkpoint_selection_to_api(document, train_config, objective)
    if result != document:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit worker checkpoint selection metadata is invalid",
        )
    return result


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} keys must be strings")
    return cast(JsonObject, dict(mapping))


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _attempt_id(job: ExecutionJobRecord) -> str:
    if job.attempt_id is None:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "worker attempt identity is unavailable",
        )
    return job.attempt_id


def _model_label(job: ExecutionJobRecord) -> str:
    if job.model_label is None:
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit model label is unavailable",
        )
    return job.model_label
