from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from app.contracts.worker.v1 import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    validate_document,
)
from app.contracts.worker.v1.config import (
    CHECKPOINT_FORMAT,
    ModelConfig,
    TrainConfig,
    model_config_to_manifest,
    train_config_to_manifest,
)
from app.worker.application.events import WorkerEventEmitter
from app.worker.data.arrow import read_arrow, read_source_arrow, write_arrow
from app.worker.data.tensors import (
    reshape_source,
    validate_checkpoint_feature_dim,
    validate_feature_dim,
    validate_target_dim,
)
from app.worker.metrics import reset_metrics_log
from app.worker.runtime.checkpoints.checkpoint import load_checkpoint
from app.worker.runtime.checkpoints.training_recovery import (
    load_training_recovery,
    save_training_recovery,
)
from app.worker.runtime.device import get_device
from app.worker.runtime.reproducibility import configure_reproducibility
from app.worker.runtime.version import __version__
from app.worker.training.factory import build_model, build_trainer

_COPY_CHUNK_BYTES = 1024 * 1024


class WorkerApplication:
    """Execute one immutable worker-v1 command manifest."""

    def __init__(self, emitter: WorkerEventEmitter):
        self.emitter = emitter

    def run(self, manifest: dict) -> None:
        validate_document(manifest, "command-manifest")
        self._validate_identity(manifest)
        workspace = _validate_workspace(manifest["workspace"]["root"])
        self.emitter.ready()
        if manifest["operation"] == "fit":
            result = self._fit(manifest, workspace)
        else:
            result = self._predict(manifest, workspace)
        result_path = os.path.join(workspace, "worker-result.json")
        _write_json_once(result_path, result)
        artifact = _artifact(result_path)
        self.emitter.completed(artifact)

    def _validate_identity(self, manifest: dict) -> None:
        expected = (
            self.emitter.job_id,
            self.emitter.attempt,
            self.emitter.attempt_id,
        )
        actual = (
            manifest["jobId"],
            manifest["attempt"],
            manifest["attemptId"],
        )
        if actual != expected:
            raise ValueError("worker command identity does not match argv")

    def _fit(self, manifest: dict, workspace: str) -> dict:
        model_config = ModelConfig.from_dict(manifest["model"]["config"])
        train_config = TrainConfig.from_dict(manifest["training"])
        if model_config is None or train_config is None:
            raise ValueError("fit configuration is unavailable")
        configure_reproducibility(
            train_config.seed,
            train_config.deterministic,
        )
        device = get_device(manifest["device"]["kind"])
        model = None
        trainer = None
        expected_feature_dim = None
        expected_target_dim = None
        input_paths: list[str] = []

        for item in manifest["inputs"]:
            if item["schemaId"] != FIT_INPUT_SCHEMA_ID:
                raise ValueError("fit input schemaId is invalid")
            path = _validate_artifact(item["artifact"])
            X_cpu, Y_cpu = read_arrow(path)
            if X_cpu.size(0) != item["rows"]:
                raise ValueError("fit input row count differs from its manifest")
            if X_cpu.size(0) == 0:
                continue
            X_cpu = reshape_source(X_cpu, model_config.seq_len)
            expected_feature_dim = validate_feature_dim(
                X_cpu,
                expected_feature_dim,
            )
            expected_target_dim = validate_target_dim(
                Y_cpu,
                expected_target_dim,
            )
            if model is None:
                actual_config = replace(
                    model_config,
                    feature_dim=expected_feature_dim,
                )
                model = build_model(actual_config, X_cpu, Y_cpu, device)
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
                )
                reset_metrics_log(metrics_path)
            input_paths.append(path)
            del X_cpu, Y_cpu

        if trainer is None:
            raise ValueError("fit requires at least one non-empty input")

        recovery = manifest.get("recovery")
        if recovery is not None and recovery.get("checkpoint") is not None:
            checkpoint_path = _validate_artifact(recovery["checkpoint"])
            try:
                payload = load_training_recovery(
                    checkpoint_path,
                    device,
                    expected_config_hash=recovery["configSha256"],
                    expected_seal_hash=recovery["manifestSha256"],
                )
                if payload["model_config"] != trainer.model_config.to_dict():
                    raise ValueError("recovery model configuration differs")
                if payload["train_config"] != train_config.to_dict():
                    raise ValueError("recovery training configuration differs")
                trainer.load_recovery_state_dict(payload["trainer_state"])
            except Exception as exc:
                raise WorkerExecutionError(
                    "RECOVERY_CHECKPOINT_INCOMPATIBLE",
                    "training recovery checkpoint could not be restored",
                ) from exc

        def payloads():
            for path in input_paths:
                X_cpu, Y_cpu = read_arrow(path)
                X_cpu = reshape_source(X_cpu, trainer.model_config.seq_len)
                validate_feature_dim(X_cpu, expected_feature_dim)
                validate_target_dim(Y_cpu, expected_target_dim)
                yield X_cpu, Y_cpu
                del X_cpu, Y_cpu

        trained_epochs = 0

        def on_epoch(epoch, metrics, monitor_payload):
            nonlocal trained_epochs
            trained_epochs += 1
            progress = metrics.to_dict(
                **trainer.metrics_context,
                mode="fit-stream",
                epoch=epoch + 1,
                **monitor_payload,
            )
            trainer.record_metrics(
                metrics,
                mode="fit-stream",
                epoch=epoch + 1,
                **monitor_payload,
            )
            self.emitter.progress(_json_safe(progress))

        def on_epoch_committed(
            _epoch,
            _metrics,
            _monitor_payload,
            _training_complete,
        ):
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
                config_hash=recovery["configSha256"],
                seal_hash=recovery["manifestSha256"],
            )
            self.emitter.checkpoint({
                "generation": event["generation"],
                "completedEpochs": event["completed_epochs"],
                "globalStep": event["global_step"],
                "trainingComplete": event["training_complete"],
                "artifact": _artifact(checkpoint_path),
            })

        if not trainer.training_complete:
            if recovery is None:
                trainer.fit_payloads(payloads, on_epoch=on_epoch)
            else:
                trainer.fit_payloads_resumable(
                    payloads,
                    on_epoch=on_epoch,
                    on_epoch_committed=on_epoch_committed,
                )

        checkpoint_path = os.path.join(workspace, "checkpoint.pth")
        trainer.save(checkpoint_path)
        print(
            f"Model saved after {len(input_paths)} trained frame(s), "
            f"{trained_epochs} epoch(s) from "
            f"{len(manifest['inputs'])} received frame(s)"
        )
        result = _result_identity(manifest)
        result.update({
            "artifacts": [],
            "checkpoint": _artifact(checkpoint_path),
            "metrics": _artifact(os.path.join(workspace, "metrics.jsonl")),
            "checkpointMetadata": _checkpoint_metadata(trainer),
        })
        validate_document(result, "result-manifest")
        return result

    def _predict(self, manifest: dict, workspace: str) -> dict:
        checkpoint_path = _validate_artifact(
            manifest["model"]["checkpoint"]
        )
        device = get_device(manifest["device"]["kind"])
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("prediction checkpoint format is unsupported")
        model_config = ModelConfig.from_dict(checkpoint.get("model_config"))
        expected_config = ModelConfig.from_dict(manifest["model"]["config"])
        if model_config is None or model_config != expected_config:
            raise ValueError(
                "prediction checkpoint configuration differs from its manifest"
            )
        train_config = TrainConfig.from_dict(checkpoint.get("train_config"))
        train_config = train_config or TrainConfig()
        prediction_column = manifest["predictionColumn"]
        model = None
        trainer = None
        expected_feature_dim = None
        artifacts = []
        received_frames = 0
        predicted_frames = 0
        for item in manifest["inputs"]:
            received_frames += 1
            if item["schemaId"] != PREDICT_INPUT_SCHEMA_ID:
                raise ValueError("prediction input schemaId is invalid")
            input_path = _validate_artifact(item["artifact"])
            X_cpu = read_source_arrow(input_path)
            if X_cpu.size(0) != item["rows"]:
                raise ValueError(
                    "prediction input row count differs from its manifest"
                )
            output_path = os.path.join(
                workspace,
                "outputs",
                f"{item['ordinal']}.arrow",
            )
            empty_input = X_cpu.size(0) == 0
            if empty_input:
                predictions = torch.empty((0, 6), dtype=torch.float32)
            else:
                X_cpu = reshape_source(X_cpu, model_config.seq_len)
                validate_checkpoint_feature_dim(
                    X_cpu,
                    model_config.feature_dim,
                )
                expected_feature_dim = validate_feature_dim(
                    X_cpu,
                    expected_feature_dim,
                )
                if model is None:
                    model = build_model(model_config, X_cpu, None, device)
                    trainer = build_trainer(
                        train_config,
                        model,
                        device,
                        model_config,
                    )
                    trainer.load_payload(checkpoint)
                    checkpoint = None
                    print("X:", X_cpu.shape)
                    print("Model loaded")
                predictions = trainer.predict(X_cpu)
            write_arrow(
                output_path,
                predictions,
                prediction_column,
                expected_rows=item["rows"],
            )
            if empty_input:
                print(
                    f"frame {received_frames}, emitted empty predictions"
                )
            else:
                predicted_frames += 1
                print(
                    f"frame {received_frames}, predicted "
                    f"{predictions.shape[0]} row(s)"
                )
            artifacts.append({
                "schemaId": PREDICTION_OUTPUT_SCHEMA_ID,
                "ordinal": item["ordinal"],
                "rows": item["rows"],
                "artifact": _artifact(output_path),
            })
            self.emitter.progress({
                "ordinal": item["ordinal"],
                "rows": item["rows"],
            })
        print(
            f"Predicted {predicted_frames} non-empty frame(s) "
            f"from {received_frames} received frame(s)"
        )
        result = _result_identity(manifest)
        result["artifacts"] = artifacts
        validate_document(result, "result-manifest")
        return result


class WorkerExecutionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _result_identity(manifest: dict) -> dict:
    return {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "jobId": manifest["jobId"],
        "attempt": manifest["attempt"],
        "attemptId": manifest["attemptId"],
        "operation": manifest["operation"],
    }


def _checkpoint_metadata(trainer) -> dict:
    model_config = trainer.model_config
    train_config = trainer.train_config
    feature_dim = model_config.feature_dim
    if feature_dim is None:
        raise ValueError("fit checkpoint feature dimension is unavailable")
    best_monitor = trainer.best_monitor
    if not isinstance(best_monitor, (int, float)) or not math.isfinite(
        best_monitor
    ):
        best_monitor = None
    return {
        "format": CHECKPOINT_FORMAT,
        "serviceVersion": __version__,
        "modelConfig": model_config_to_manifest(model_config),
        "trainingConfig": train_config_to_manifest(train_config),
        "dataSchema": {
            "schemaVersion": 1,
            "tensorDtype": "float32",
            "source": {
                "column": "src",
                "acceptedElementTypes": ["float32", "float64"],
                "width": model_config.seq_len * feature_dim,
            },
            "target": {
                "column": "tgt",
                "acceptedElementTypes": ["float32", "float64"],
                "width": 6,
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
        },
        "checkpointSelection": {
            "monitor": trainer.monitor,
            "monitorMinImprovement": trainer.monitor_min_improvement,
            "bestMonitor": best_monitor,
            "bestFrame": trainer.best_frame,
            "bestEpoch": trainer.best_epoch,
            "baselinePassed": trainer.best_state_dict is not None,
            "source": (
                "best_monitor"
                if trainer.best_state_dict is not None
                else "current"
            ),
        },
    }


def _validate_workspace(path: str) -> str:
    workspace = os.path.abspath(os.fspath(path))
    if not os.path.isabs(path) or os.path.realpath(workspace) != workspace:
        raise ValueError("worker workspace must be a canonical managed path")
    if not os.path.isdir(workspace):
        raise ValueError("worker workspace is unavailable")
    return workspace


def _validate_artifact(document: dict) -> str:
    path = os.path.abspath(os.fspath(document["path"]))
    if not os.path.isabs(document["path"]):
        raise ValueError("worker artifact path must be absolute")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ValueError("worker input artifact is unavailable") from exc
    if size != document["byteCount"] or _sha256_file(path) != document["sha256"]:
        raise ValueError("worker input artifact integrity check failed")
    return path


def _artifact(path: str) -> dict:
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


def _write_json_once(path: str, document: dict) -> None:
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


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


__all__ = ["WorkerApplication", "WorkerExecutionError"]
