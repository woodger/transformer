from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import torch

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v3 import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    validate_document,
)
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
)
from app.worker.application.documents import (
    integer_field,
    object_document,
    object_field,
    optional_string_field,
    string_field,
)
from app.worker.application.events import WorkerEventEmitter
from app.worker.application.inputs import DurableInputStream
from app.worker.data.arrow import (
    read_committed_fit_arrow,
    read_committed_source_arrow,
    write_arrow,
)
from app.worker.data.tensors import (
    reshape_source,
    validate_checkpoint_feature_dim,
    validate_feature_dim,
    validate_target_dim,
)
from app.worker.metrics import TrainMetrics, reset_metrics_log
from app.worker.runtime.checkpoints.checkpoint import (
    CheckpointCorrupt,
    CheckpointFormatMismatch,
    load_checkpoint,
)
from app.worker.runtime.checkpoints.training_recovery import (
    load_training_recovery,
    save_training_recovery,
)
from app.worker.runtime.device import get_device
from app.worker.runtime.reproducibility import configure_reproducibility
from app.worker.runtime.version import __version__
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import (
    SelectionPayload,
    Trainer,
)

_COPY_CHUNK_BYTES = 1024 * 1024


class WorkerApplication:
    """Execute one durable-streaming worker-v3 command manifest."""

    def __init__(
        self,
        emitter: WorkerEventEmitter,
        input_stream: BinaryIO | None = None,
    ) -> None:
        self.emitter = emitter
        self.input_stream = input_stream

    def run(self, manifest: JsonObject) -> None:
        validate_document(manifest, "command-manifest")
        self._validate_identity(manifest)
        workspace = _validate_workspace(
            string_field(object_field(manifest, "workspace"), "root")
        )
        if self.input_stream is None:
            raise ValueError("worker control stream is unavailable")
        inputs = DurableInputStream(
            manifest,
            self.input_stream,
            self.emitter,
        )
        self.emitter.ready(
            next_ordinal=inputs.next_ordinal,
            input_revision=inputs.input_revision,
        )
        if string_field(manifest, "operation") == "fit":
            result = self._fit_streaming(manifest, workspace, inputs)
        else:
            result = self._predict_streaming(manifest, workspace, inputs)
        result_path = os.path.join(workspace, "worker-result.json")
        _write_json_once(result_path, result)
        artifact = _artifact(result_path)
        self.emitter.completed(artifact)

    def _validate_identity(self, manifest: JsonObject) -> None:
        expected = (
            self.emitter.job_id,
            self.emitter.attempt,
            self.emitter.attempt_id,
        )
        actual = (
            string_field(manifest, "jobId"),
            integer_field(manifest, "attempt"),
            string_field(manifest, "attemptId"),
        )
        if actual != expected:
            raise ValueError("worker command identity does not match argv")

    def _fit_streaming(
        self,
        manifest: JsonObject,
        workspace: str,
        input_stream: DurableInputStream,
    ) -> JsonObject:
        model_document = object_field(manifest, "model")
        model_config = ModelConfig.from_dict(
            object_field(model_document, "config")
        )
        train_config = TrainConfig.from_dict(
            object_field(manifest, "training")
        )
        if model_config is None or train_config is None:
            raise ValueError("fit configuration is unavailable")
        if object_field(manifest, "mlContract") != ml_contract(train_config):
            raise ValueError("fit ML contract differs from training configuration")
        configure_reproducibility(
            train_config.seed,
            train_config.deterministic,
        )
        device = get_device(
            string_field(object_field(manifest, "device"), "kind")
        )
        data_contract = object_field(manifest, "dataContract")
        expected_feature_dim = integer_field(data_contract, "featureDim")
        expected_target_dim = 6
        committed_inputs = _CommittedInputArtifacts()

        def read_payload(
            item: JsonObject,
        ) -> tuple[str, torch.Tensor, torch.Tensor]:
            if string_field(item, "schemaId") != FIT_INPUT_SCHEMA_ID:
                raise ValueError("fit input schemaId is invalid")
            path = committed_inputs.path(item)
            source, target = read_committed_fit_arrow(
                path,
                expected_rows=integer_field(item, "rows"),
                source_width=model_config.seq_len * expected_feature_dim,
            )
            source = reshape_source(source, model_config.seq_len)
            validate_feature_dim(source, expected_feature_dim)
            validate_target_dim(target, expected_target_dim)
            return path, source, target

        stream = iter(input_stream.items())
        first = None
        for item in stream:
            path, source, target = read_payload(item)
            if source.size(0) == 0:
                continue
            first = (path, source, target)
            break
        if first is None:
            raise ValueError("fit requires at least one non-empty input")

        actual_config = replace(
            model_config,
            feature_dim=expected_feature_dim,
        )
        model = build_model(actual_config, first[1], first[2], device)
        metrics_path = os.path.join(workspace, "metrics.jsonl")
        trainer_args = SimpleNamespace(
            **train_config.to_dict(),
            metrics_name=metrics_path,
        )
        trainer = build_trainer(
            trainer_args,
            model,
            device,
            actual_config,
            data_contract=data_contract,
        )
        reset_metrics_log(metrics_path)

        recovery_value = manifest.get("recovery")
        recovery = (
            None
            if recovery_value is None
            else object_document(recovery_value, "recovery")
        )
        if recovery is not None and recovery.get("checkpoint") is not None:
            if (
                not input_stream.closed
                or optional_string_field(recovery, "manifestSha256") is None
            ):
                raise ValueError(
                    "recovery checkpoint requires closed immutable input"
                )
            checkpoint_path = _validate_artifact(
                object_field(recovery, "checkpoint")
            )
            try:
                payload = load_training_recovery(
                    checkpoint_path,
                    device,
                    expected_config_hash=string_field(
                        recovery,
                        "configSha256",
                    ),
                    expected_manifest_hash=string_field(
                        recovery,
                        "manifestSha256",
                    ),
                    expected_objective_config_sha256=string_field(
                        recovery,
                        "objectiveConfigSha256",
                    ),
                    expected_data_contract_sha256=string_field(
                        recovery,
                        "dataContractSha256",
                    ),
                )
                if (
                    trainer.model_config is None
                    or payload["model_config"]
                    != trainer.model_config.to_dict()
                ):
                    raise ValueError("recovery model configuration differs")
                if payload["train_config"] != train_config.to_dict():
                    raise ValueError("recovery training configuration differs")
                trainer.load_recovery_state_dict(
                    object_document(
                        payload["trainer_state"],
                        "trainer state",
                    )
                )
            except Exception as exc:
                raise WorkerExecutionError(
                    "RECOVERY_CHECKPOINT_INCOMPATIBLE",
                    "training recovery checkpoint could not be restored",
                ) from exc

        def first_epoch_payloads() -> Iterator[
            tuple[torch.Tensor, torch.Tensor]
        ]:
            yield first[1], first[2]
            for item in stream:
                _path, source, target = read_payload(item)
                if source.size(0) != 0:
                    yield source, target

        def closed_payloads() -> Iterator[
            tuple[torch.Tensor, torch.Tensor]
        ]:
            if not input_stream.closed:
                raise ValueError("complete input is unavailable for replay")
            for item in input_stream.inputs:
                _path, source, target = read_payload(item)
                if source.size(0) != 0:
                    yield source, target

        def on_epoch(
            epoch: int,
            metrics: TrainMetrics,
            monitor_payload: SelectionPayload,
        ) -> None:
            progress = metrics.to_dict(
                **trainer.metrics_context,
                mode="fit-stream",
                epoch=epoch + 1,
                selection_score=monitor_payload["selection_score"],
                checkpoint_best=monitor_payload["checkpoint_best"],
                should_stop=monitor_payload["should_stop"],
                best_selection_score=monitor_payload[
                    "best_selection_score"
                ],
            )
            trainer.record_metrics(
                metrics,
                mode="fit-stream",
                epoch=epoch + 1,
                selection_score=monitor_payload["selection_score"],
                checkpoint_best=monitor_payload["checkpoint_best"],
                should_stop=monitor_payload["should_stop"],
                best_selection_score=monitor_payload[
                    "best_selection_score"
                ],
            )
            self.emitter.progress(
                object_document(_json_safe(progress), "fit progress")
            )

        def on_epoch_committed(
            _epoch: int,
            _metrics: TrainMetrics,
            _monitor_payload: SelectionPayload,
            _training_complete: bool,
        ) -> None:
            if recovery is None:
                return
            manifest_sha256 = input_stream.manifest_sha256
            if manifest_sha256 is None:
                raise ValueError(
                    "recovery checkpoint cannot precede input EOF"
                )
            generation = trainer.state.global_epoch
            checkpoint_path = os.path.join(
                workspace,
                "checkpoints",
                f"{generation}.pth",
            )
            event = save_training_recovery(
                checkpoint_path,
                trainer,
                generation=generation,
                config_hash=string_field(recovery, "configSha256"),
                manifest_hash=manifest_sha256,
            )
            self.emitter.checkpoint({
                "generation": integer_field(event, "generation"),
                "completedEpochs": integer_field(event, "completed_epochs"),
                "globalStep": integer_field(event, "global_step"),
                "trainingComplete": _boolean_value(
                    event.get("training_complete"),
                    "training_complete",
                ),
                "artifact": _artifact(checkpoint_path),
            })

        if not trainer.training_complete:
            if trainer.state.global_epoch == 0:
                trainer.fit_streaming_payloads(
                    first_epoch_payloads(),
                    closed_payloads,
                    on_epoch=on_epoch,
                    on_epoch_committed=on_epoch_committed,
                )
            else:
                trainer.fit_payloads_resumable(
                    closed_payloads,
                    on_epoch=on_epoch,
                    on_epoch_committed=on_epoch_committed,
                )

        checkpoint_path = os.path.join(workspace, "checkpoint.pth")
        trainer.save(checkpoint_path)
        result = _result_identity(manifest)
        result.update({
            "inputRevision": input_stream.input_revision,
            "manifestSha256": input_stream.manifest_sha256,
            "artifacts": [],
            "checkpoint": _artifact(checkpoint_path),
            "metrics": _artifact(metrics_path),
            "checkpointMetadata": _checkpoint_metadata(
                trainer,
                data_contract,
            ),
        })
        validate_document(result, "result-manifest")
        return result

    def _predict_streaming(
        self,
        manifest: JsonObject,
        workspace: str,
        input_stream: DurableInputStream,
    ) -> JsonObject:
        model_document = object_field(manifest, "model")
        checkpoint_path = _validate_artifact(
            object_field(model_document, "checkpoint")
        )
        device = get_device(
            string_field(object_field(manifest, "device"), "kind")
        )
        try:
            checkpoint = load_checkpoint(checkpoint_path, device)
        except CheckpointFormatMismatch as exc:
            raise WorkerExecutionError(
                "MODEL_SCHEMA_MISMATCH",
                "prediction checkpoint belongs to another ML contract",
            ) from exc
        except CheckpointCorrupt as exc:
            raise WorkerExecutionError(
                "MODEL_CORRUPT",
                "prediction checkpoint semantic metadata is invalid",
            ) from exc
        model_config = ModelConfig.from_dict(checkpoint.get("model_config"))
        expected_config = ModelConfig.from_dict(
            object_field(model_document, "config")
        )
        if model_config is None or model_config != expected_config:
            raise ValueError(
                "prediction checkpoint configuration differs from its manifest"
            )
        train_config = TrainConfig.from_dict(checkpoint["train_config"])
        if train_config is None:
            raise WorkerExecutionError(
                "MODEL_CORRUPT",
                "prediction checkpoint has no training configuration",
            )
        if checkpoint["data_contract"] is None:
            raise WorkerExecutionError(
                "MODEL_CORRUPT",
                "prediction checkpoint has no data contract",
            )
        if (
            checkpoint["data_contract"] != object_field(manifest, "dataContract")
            or checkpoint["ml_contract"] != object_field(manifest, "mlContract")
            or checkpoint["ml_contract"] != ml_contract(train_config)
        ):
            raise WorkerExecutionError(
                "MODEL_SCHEMA_MISMATCH",
                "prediction checkpoint contract differs from the job",
            )
        if model_config.feature_dim is None:
            raise WorkerExecutionError(
                "MODEL_CORRUPT",
                "prediction checkpoint feature dimension is unavailable",
            )
        prediction_column = string_field(manifest, "predictionColumn")
        model: torch.nn.Module | None = None
        trainer: Trainer | None = None
        artifacts: list[JsonValue] = []
        data_contract = object_field(manifest, "dataContract")
        committed_inputs = _CommittedInputArtifacts()
        for item in input_stream.items():
            if string_field(item, "schemaId") != PREDICT_INPUT_SCHEMA_ID:
                raise ValueError("prediction input schemaId is invalid")
            input_path = committed_inputs.path(item)
            source = read_committed_source_arrow(
                input_path,
                expected_rows=integer_field(item, "rows"),
                source_width=model_config.seq_len * model_config.feature_dim,
            )
            output_path = os.path.join(
                workspace,
                "outputs",
                f"{integer_field(item, 'ordinal')}.arrow",
            )
            if source.size(0) == 0:
                predictions = torch.empty((0, 6), dtype=torch.float32)
            else:
                source = reshape_source(source, model_config.seq_len)
                validate_checkpoint_feature_dim(
                    source,
                    model_config.feature_dim,
                )
                if model is None:
                    model = build_model(model_config, source, None, device)
                    trainer = build_trainer(
                        train_config,
                        model,
                        device,
                        model_config,
                        data_contract=data_contract,
                    )
                    if checkpoint is None:
                        raise AssertionError(
                            "prediction checkpoint was already consumed"
                        )
                    trainer.load_payload(checkpoint)
                    checkpoint = None
                if trainer is None:
                    raise AssertionError("prediction trainer was not initialized")
                predictions = trainer.predict(source)
            write_arrow(
                output_path,
                predictions,
                prediction_column,
                expected_rows=integer_field(item, "rows"),
            )
            artifacts.append({
                "schemaId": PREDICTION_OUTPUT_SCHEMA_ID,
                "ordinal": integer_field(item, "ordinal"),
                "commitRevision": integer_field(item, "commitRevision"),
                "dataContractSha256": string_field(
                    item,
                    "dataContractSha256",
                ),
                "rows": integer_field(item, "rows"),
                "artifact": _artifact(output_path),
            })
            self.emitter.progress({
                "ordinal": integer_field(item, "ordinal"),
                "rows": integer_field(item, "rows"),
            })
        result = _result_identity(manifest)
        result["inputRevision"] = input_stream.input_revision
        result["manifestSha256"] = input_stream.manifest_sha256
        result["artifacts"] = artifacts
        validate_document(result, "result-manifest")
        return result

class WorkerExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _CommittedInputArtifacts:
    """Verify each immutable input receipt exactly once per worker attempt."""

    def __init__(self) -> None:
        self._verified: dict[
            int,
            tuple[tuple[object, ...], str],
        ] = {}

    def path(self, item: JsonObject) -> str:
        ordinal = integer_field(item, "ordinal")
        identity = _input_receipt_identity(item)
        existing = self._verified.get(ordinal)
        if existing is None:
            path = _validate_artifact(object_field(item, "artifact"))
            self._verified[ordinal] = (identity, path)
            return path
        if existing[0] != identity:
            raise ValueError(
                "committed input receipt changed during the worker attempt"
            )
        return existing[1]


def _input_receipt_identity(item: JsonObject) -> tuple[object, ...]:
    artifact = object_field(item, "artifact")
    return (
        string_field(item, "schemaId"),
        integer_field(item, "ordinal"),
        integer_field(item, "commitRevision"),
        string_field(item, "dataContractSha256"),
        integer_field(item, "rows"),
        string_field(artifact, "path"),
        integer_field(artifact, "byteCount"),
        string_field(artifact, "sha256"),
    )


def _result_identity(manifest: JsonObject) -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "jobId": string_field(manifest, "jobId"),
        "attempt": integer_field(manifest, "attempt"),
        "attemptId": string_field(manifest, "attemptId"),
        "operation": string_field(manifest, "operation"),
    }


def _checkpoint_metadata(
    trainer: Trainer,
    data_contract: JsonObject | None = None,
) -> JsonObject:
    model_config = trainer.model_config
    train_config = trainer.train_config
    if model_config is None:
        raise ValueError("fit checkpoint configuration is unavailable")
    feature_dim = model_config.feature_dim
    if feature_dim is None:
        raise ValueError("fit checkpoint feature dimension is unavailable")
    if data_contract is None:
        raise ValueError("fit checkpoint data contract is unavailable")
    best_selection_score = trainer.best_selection_score
    if not math.isfinite(best_selection_score):
        best_selection_score = None
    selection_enabled = trainer.selection is not None
    result: JsonObject = {
        "format": CHECKPOINT_FORMAT,
        "serviceVersion": __version__,
        "modelConfig": model_config_to_manifest(model_config),
        "trainingConfig": train_config_to_manifest(train_config),
        "dataContract": dict(data_contract),
        "mlContract": ml_contract(train_config),
        "objectiveConfig": objective_config(train_config),
        "checkpointSelection": {
            "enabled": selection_enabled,
            "objectiveConfigSha256": ml_contract(train_config)[
                "objectiveConfigSha256"
            ],
            "bestSelectionScore": best_selection_score,
            "bestFrame": trainer.best_frame,
            "bestEpoch": trainer.best_epoch,
            "source": (
                "best_selection_score"
                if selection_enabled
                else "last_maximum_stage"
            ),
        },
    }
    return result


def _validate_workspace(path: str) -> str:
    workspace = os.path.abspath(os.fspath(path))
    if not os.path.isabs(path) or os.path.realpath(workspace) != workspace:
        raise ValueError("worker workspace must be a canonical managed path")
    if not os.path.isdir(workspace):
        raise ValueError("worker workspace is unavailable")
    return workspace


def _validate_artifact(document: JsonObject) -> str:
    documented_path = string_field(document, "path")
    path = os.path.abspath(documented_path)
    if not os.path.isabs(documented_path):
        raise ValueError("worker artifact path must be absolute")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ValueError("worker input artifact is unavailable") from exc
    if (
        size != integer_field(document, "byteCount")
        or _sha256_file(path) != string_field(document, "sha256")
    ):
        raise ValueError("worker input artifact integrity check failed")
    return path


def _artifact(path: str) -> JsonObject:
    path = os.path.abspath(path)
    return {
        "path": path,
        "byteCount": os.path.getsize(path),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: str, document: JsonObject) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=parent,
        prefix=f".{Path(path).name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("worker result manifest already exists") from exc
        _fsync_directory(parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _fsync_directory(path: str) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("worker JSON field names must be strings")
        return {
            cast(str, key): _json_safe(item)
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in cast(Sequence[object], value)]
    raise TypeError(f"worker value is not JSON-compatible: {type(value).__name__}")


def _boolean_value(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"worker {label} must be a boolean")
    return value


__all__ = ["WorkerApplication", "WorkerExecutionError"]
