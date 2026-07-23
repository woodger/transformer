from __future__ import annotations

import errno
import hashlib
import math
import os
import shutil
import uuid
from dataclasses import dataclass

from app.flight.arrow import validate_prediction_file
from app.flight.constants import ErrorCode, JobState
from app.flight.contract import model_config_to_api, train_config_to_api
from app.flight.errors import ServiceError
from app.flight.records import ExecutionJobRecord
from app.flight.worker_plan import ExecutionInput
from app.storage.checkpoint import CHECKPOINT_FORMAT, load_checkpoint_metadata
from app.training.run_config import ModelConfig, TrainConfig

_COPY_CHUNK_BYTES = 1024 * 1024
_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}


@dataclass(frozen=True)
class WorkerArtifactError(Exception):
    code: ErrorCode
    message: str

    def __post_init__(self):
        Exception.__init__(self, self.message)


@dataclass(frozen=True)
class StagedPredictionOutput:
    ordinal: int
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str

    def ledger_record(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "rows": self.rows,
            "batches": self.batches,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "schema_fingerprint": self.schema_fingerprint,
            "relative_path": self.relative_path,
        }


class WorkerArtifactPublisher:
    """Stage, validate, publish, and clean up worker-owned artifacts."""

    def __init__(self, ledger, spool, *, logger, metrics):
        self.ledger = ledger
        self.spool = spool
        self.logger = logger
        self.metrics = metrics

    def stage_prediction(
        self,
        stream,
        job: ExecutionJobRecord,
        item: ExecutionInput,
        size: int,
    ) -> StagedPredictionOutput:
        destination = self.spool.attempt_output_path(
            job.job_id,
            job.attempt,
            item.ordinal,
        )
        try:
            return self._stage_prediction(
                stream,
                destination,
                size,
                job.prediction_column,
                item.rows,
                item.ordinal,
            )
        except OSError as exc:
            if exc.errno in _DISK_FULL_ERRNOS:
                raise WorkerArtifactError(
                    ErrorCode.DISK_FULL,
                    "prediction output could not be persisted",
                ) from exc
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess emitted malformed Arrow output",
            ) from exc
        except (EOFError, ServiceError, ValueError) as exc:
            raise WorkerArtifactError(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess emitted malformed Arrow output",
            ) from exc

    def publish_outputs(
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

    def publish_model(self, job: ExecutionJobRecord) -> None:
        attempt_path = self.spool.attempt_checkpoint_path(
            job.job_id, job.attempt
        )
        if not os.path.isfile(attempt_path):
            raise WorkerArtifactError(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess did not create a checkpoint",
            )
        try:
            checkpoint = load_checkpoint_metadata(attempt_path, "cpu")
            actual_model = ModelConfig.from_dict(checkpoint.get("model_config"))
            actual_train = TrainConfig.from_dict(checkpoint.get("train_config"))
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
        if checkpoint.get("format") != CHECKPOINT_FORMAT or actual_model is None:
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
            safe_checkpoint = {
                "format": checkpoint["format"],
                "serviceVersion": checkpoint.get("version"),
                "sha256": digest,
                "bytes": byte_count,
                "modelConfig": model_config_to_api(actual_model),
                "trainConfig": train_config_to_api(actual_train),
                "dataSchema": _data_schema_to_api(
                    checkpoint.get("data_schema"),
                    actual_model,
                ),
                "checkpointSelection": _checkpoint_selection_to_api(
                    (checkpoint.get("extra") or {}).get("checkpoint_selection"),
                    actual_train,
                ),
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
                model_ref=model_ref,
                label=job.model_label,
                generation=None,
                checkpoint_path=self.spool.model_relative_path(checkpoint_path),
                metadata_path=self.spool.model_relative_path(metadata_path),
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
            if current is None or current.state != JobState.SUCCEEDED:
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

    def _stage_prediction(
        self,
        stream,
        destination: str,
        size: int,
        prediction_column: str,
        expected_rows: int,
        ordinal: int,
    ) -> StagedPredictionOutput:
        target, temporary = self.spool.create_temporary(destination)
        digest = hashlib.sha256()
        try:
            remaining = size
            while remaining:
                chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
                if not chunk:
                    raise EOFError("incomplete prediction frame")
                target.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
            target.close()
            stats = validate_prediction_file(
                temporary,
                prediction_column,
                expected_rows,
            )
            self.spool.durable_replace(temporary, destination)
            temporary = None
            return StagedPredictionOutput(
                ordinal=ordinal,
                rows=stats.rows,
                batches=stats.batches,
                byte_count=size,
                sha256=digest.hexdigest(),
                schema_fingerprint=stats.schema_fingerprint,
                relative_path=self.spool.relative_path(destination),
            )
        finally:
            if not target.closed:
                target.close()
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_schema_to_api(value: dict | None, model_config: ModelConfig) -> dict:
    if not isinstance(value, dict):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint does not contain data schema metadata",
        )
    source = value.get("src")
    target = value.get("tgt")
    missing = value.get("missing")
    if not all(isinstance(item, dict) for item in (source, target, missing)):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid data schema metadata",
        )
    result = {
        "schemaVersion": value.get("schema_version"),
        "tensorDtype": value.get("tensor_dtype"),
        "source": {
            "column": source.get("column"),
            "acceptedElementTypes": source.get("accepted_element_types"),
            "width": source.get("width"),
        },
        "target": {
            "column": target.get("column"),
            "acceptedElementTypes": target.get("accepted_element_types"),
            "width": target.get("width"),
        },
        "featureDim": value.get("feature_dim"),
        "modelInputFeatureDim": value.get("model_input_feature_dim"),
        "contextMode": value.get("context_mode"),
        "normalization": value.get("normalization"),
        "missing": {
            "nanFill": missing.get("nan_fill"),
            "flags": missing.get("flags"),
        },
    }
    expected_types = ["float32", "float64"]
    expected_input_dim = (
        model_config.feature_dim * 2
        if model_config.context_mode == "relaxed"
        else model_config.feature_dim
    )
    if (
        result["schemaVersion"] != 1
        or result["tensorDtype"] != "float32"
        or result["source"]["column"] != "src"
        or result["source"]["acceptedElementTypes"] != expected_types
        or type(result["source"]["width"]) is not int
        or result["source"]["width"] <= 0
        or result["target"]["column"] != "tgt"
        or result["target"]["acceptedElementTypes"] != expected_types
        or result["target"]["width"] != 6
        or result["featureDim"] != model_config.feature_dim
        or result["source"]["width"]
        != model_config.seq_len * model_config.feature_dim
        or result["modelInputFeatureDim"] != expected_input_dim
        or result["contextMode"] != model_config.context_mode
        or result["normalization"] is not None
        or result["missing"]["nanFill"] != 0.0
        or result["missing"]["flags"]
        != ("per-feature" if model_config.context_mode == "relaxed" else "none")
    ):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid data schema metadata",
        )
    return result


def _checkpoint_selection_to_api(
    value: dict | None,
    train_config: TrainConfig,
) -> dict:
    value = value if isinstance(value, dict) else {}
    result = {
        "monitor": value.get("monitor", train_config.monitor),
        "monitorMinImprovement": value.get(
            "monitor_min_improvement",
            train_config.monitor_min_improvement,
        ),
        "bestMonitor": value.get("best_monitor"),
        "bestFrame": value.get("best_frame"),
        "bestEpoch": value.get("best_epoch"),
        "baselinePassed": bool(value.get("baseline_passed", False)),
        "source": value.get("source", "current"),
    }
    best_monitor = result["bestMonitor"]
    best_frame = result["bestFrame"]
    best_epoch = result["bestEpoch"]
    if (
        result["monitor"] != train_config.monitor
        or result["monitorMinImprovement"] != train_config.monitor_min_improvement
        or (
            best_monitor is not None
            and (
                isinstance(best_monitor, bool)
                or not isinstance(best_monitor, (int, float))
                or not math.isfinite(best_monitor)
            )
        )
        or (
            best_frame is not None
            and (type(best_frame) is not int or best_frame < 0)
        )
        or (
            best_epoch is not None
            and (type(best_epoch) is not int or best_epoch < 0)
        )
        or (
            "baseline_passed" in value
            and not isinstance(value["baseline_passed"], bool)
        )
        or result["source"] not in ("best_monitor", "current")
    ):
        raise WorkerArtifactError(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        )
    return result
