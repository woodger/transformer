from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import replace

import torch

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v4 import ModelContract
from app.contracts.target_head_diagnostics.v3.constants import (
    MAX_COMMITTED_ARTIFACT_ROWS,
)
from app.contracts.worker.v18 import (
    FIT_INPUT_SCHEMA_ID,
    validate_document,
    validate_training_metrics_for_model,
)
from app.contracts.worker.v18.config import ModelConfig, TrainConfig
from app.contracts.worker.v18.diagnostics import DiagnosticsConfig
from app.contracts.worker.v18.model_definition import resolved_semantic_digests
from app.worker.application.artifacts import (
    CommittedInputArtifacts,
    checkpoint_artifact_document,
    checkpoint_metadata,
    json_safe,
    result_identity,
    validate_checkpoint_artifact,
)
from app.worker.application.documents import (
    integer_field,
    integer_list,
    object_document,
    object_field,
    string_field,
)
from app.worker.application.errors import WorkerExecutionError
from app.worker.application.events import WorkerEventEmitter
from app.worker.application.inputs import DurableInputStream
from app.worker.checkpoints.model import (
    CheckpointCorrupt,
    CheckpointFormatMismatch,
    load_checkpoint,
)
from app.worker.checkpoints.recovery import (
    load_training_recovery,
    save_training_recovery,
)
from app.worker.data.arrow import (
    iter_committed_fit_arrow,
    iter_committed_fit_features,
)
from app.worker.data.tensors import (
    TrainingBatch,
    validate_feature_dim,
    validate_target_dim,
)
from app.worker.runtime.device import get_device
from app.worker.runtime.reproducibility import configure_reproducibility
from app.worker.telemetry import epoch_telemetry_document
from app.worker.telemetry.epoch import ObservedTrainingEpoch
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import SelectionPayload, Trainer


def execute_fit(
    manifest: JsonObject,
    workspace: str,
    input_stream: DurableInputStream,
    emitter: WorkerEventEmitter,
) -> JsonObject:
    model_config = ModelConfig.from_manifest(
        object_field(manifest, "modelConfig")
    )
    model_contract = _validated_model_contract(manifest, model_config)
    train_config = TrainConfig.from_dict(object_field(manifest, "training"))
    if train_config is None:
        raise ValueError("fit training configuration is unavailable")
    diagnostics = DiagnosticsConfig.from_document(
        object_field(manifest, "diagnostics")
    )
    train_config = replace(train_config, diagnostics=diagnostics)
    configure_reproducibility(train_config.seed, train_config.deterministic)
    device = get_device(
        string_field(object_field(manifest, "device"), "backend")
    )
    source_encoding = object_field(manifest, "sourceEncoding")
    committed_inputs = CommittedInputArtifacts()

    def read_payload(item: JsonObject) -> Iterator[TrainingBatch]:
        if string_field(item, "schemaId") != FIT_INPUT_SCHEMA_ID:
            raise ValueError("fit input schemaId is invalid")
        yield from iter_committed_fit_arrow(
            committed_inputs.path(item),
            expected_rows=integer_field(item, "logicalRows"),
            expected_chunks=integer_field(item, "chunks"),
            expected_native_rows=integer_list(item.get("nativeRows"), "nativeRows"),
            source_encoding=source_encoding,
            seq_len=model_config.seq_len,
            feature_dim=model_config.feature_dim,
            target_contract=model_contract.target_contract,
            binary_target_indices=model_contract.weighted_binary_target_indices,
        )

    def decoded(items: Iterator[JsonObject]) -> Iterator[TrainingBatch]:
        for item in items:
            for batch in read_payload(item):
                validate_feature_dim(batch.features, model_config.feature_dim)
                validate_target_dim(batch.targets, model_contract.target_width)
                yield batch

    decoded_stream = decoded(iter(input_stream.items()))
    first = next(
        (batch for batch in decoded_stream if batch.features.size(0) != 0),
        None,
    )
    if first is None:
        raise ValueError("fit requires at least one non-empty input")

    initialization, parent_checkpoint = _load_initialization(
        manifest,
        device,
        model_contract,
    )
    model = build_model(
        model_config,
        first.features,
        first.targets,
        device,
        model_contract,
    )
    trainer = build_trainer(
        train_config,
        model,
        device,
        model_config,
        model_contract=model_contract,
        model_definition_sha256=string_field(
            object_field(manifest, "semanticDigests"),
            "modelDefinitionSha256",
        ),
        initialization=initialization,
    )
    if parent_checkpoint is not None:
        try:
            trainer.load_payload(parent_checkpoint)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise WorkerExecutionError(
                "MODEL_CORRUPT",
                "parent checkpoint model state could not be loaded",
            ) from exc

    recovery_value = manifest.get("recovery")
    recovery = (
        None
        if recovery_value is None
        else object_document(recovery_value, "recovery")
    )
    if recovery is not None:
        _restore_recovery(trainer, recovery, device, manifest)

    def first_epoch_payloads() -> Iterator[TrainingBatch]:
        yield first
        yield from (
            batch for batch in decoded_stream if batch.features.size(0) != 0
        )

    def closed_payloads() -> Iterator[TrainingBatch]:
        if not input_stream.closed:
            raise ValueError("complete input is unavailable for replay")
        for batch in decoded(iter(input_stream.inputs)):
            if batch.features.size(0) != 0:
                yield batch

    def diagnostic_features() -> Iterator[torch.Tensor]:
        if not input_stream.closed:
            raise ValueError("complete input is unavailable for diagnostics")
        for item in input_stream.inputs:
            if string_field(item, "schemaId") != FIT_INPUT_SCHEMA_ID:
                raise ValueError("fit input schemaId is invalid")
            for features in iter_committed_fit_features(
                committed_inputs.path(item),
                expected_rows=integer_field(item, "logicalRows"),
                expected_chunks=integer_field(item, "chunks"),
                expected_native_rows=integer_list(
                    item.get("nativeRows"),
                    "nativeRows",
                ),
                source_encoding=source_encoding,
                seq_len=model_config.seq_len,
                feature_dim=model_config.feature_dim,
                target_contract=model_contract.target_contract,
            ):
                validate_feature_dim(features, model_config.feature_dim)
                if features.size(0) != 0:
                    yield features

    def on_epoch_committed(
        epoch: int,
        metrics: ObservedTrainingEpoch,
        monitor_payload: SelectionPayload,
        _training_complete: bool,
    ) -> None:
        if trainer.target_head_diagnostics_enabled:
            diagnostic_rows = sum(
                integer_field(item, "logicalRows")
                for item in input_stream.inputs
            )
            if diagnostic_rows <= MAX_COMMITTED_ARTIFACT_ROWS:
                trainer.collect_target_head_diagnostics(
                    diagnostic_features,
                    epoch=epoch + 1,
                    global_step=metrics.step,
                    expected_rows=diagnostic_rows,
                )
        emitter.progress({
            "epoch": epoch + 1,
            "step": metrics.step,
            "loss": metrics.loss,
        })
        completed_manifest = _completed_manifest(manifest, input_stream)
        generation = trainer.state.global_epoch
        checkpoint_path = os.path.join(
            workspace,
            "checkpoints",
            f"{generation}.pth",
        )
        metadata = checkpoint_metadata(trainer, completed_manifest)
        serialization_started = time.monotonic()
        event = save_training_recovery(
            checkpoint_path,
            trainer,
            metadata=metadata,
        )
        serialization_ms = (time.monotonic() - serialization_started) * 1000.0
        metrics_document = _training_metrics(
            metrics,
            epoch,
            monitor_payload,
            model_contract,
        )
        checkpoint_event: JsonObject = {
            "generation": integer_field(event, "generation"),
            "progress": dict(object_field(metadata, "progress")),
            "artifact": checkpoint_artifact_document(checkpoint_path),
            "checkpointSerializationMs": serialization_ms,
        }
        if metrics_document is not None:
            checkpoint_event["metrics"] = metrics_document
        emitter.checkpoint(checkpoint_event)

    if not trainer.training_complete:
        if trainer.state.global_epoch == 0:
            trainer.fit_streaming_payloads(
                first_epoch_payloads(),
                closed_payloads,
                on_epoch_committed=on_epoch_committed,
            )
        else:
            trainer.fit_payloads_resumable(
                closed_payloads,
                on_epoch_committed=on_epoch_committed,
            )

    if not input_stream.closed:
        raise ValueError("fit result requires a closed immutable input")
    completed_manifest = _completed_manifest(manifest, input_stream)
    completed_manifest_sha256 = string_field(
        completed_manifest,
        "manifestSha256",
    )
    metadata = checkpoint_metadata(trainer, completed_manifest)
    checkpoint_path = os.path.join(workspace, "checkpoint.pth")
    serialization_started = time.monotonic()
    trainer.save(checkpoint_path, metadata=metadata)
    serialization_ms = (time.monotonic() - serialization_started) * 1000.0
    result = result_identity(manifest)
    result.update({
        "inputRevision": input_stream.input_revision,
        "manifestSha256": completed_manifest_sha256,
        "artifacts": [],
        "checkpoint": checkpoint_artifact_document(checkpoint_path),
        "checkpointMetadata": metadata,
        "checkpointSerializationMs": serialization_ms,
        "targetHeadDiagnostics": trainer.target_head_diagnostics_artifact(
            job_id=string_field(manifest, "jobId"),
            attempt=integer_field(manifest, "attempt"),
            attempt_id=string_field(manifest, "attemptId"),
            input_revision=input_stream.input_revision,
            manifest_sha256=completed_manifest_sha256,
            job_config_sha256=string_field(manifest, "jobConfigSha256"),
        ),
    })
    validate_document(result, "result-manifest")
    return result


def _validated_model_contract(
    manifest: JsonObject,
    model_config: ModelConfig,
) -> ModelContract:
    model_contract = ModelContract.from_document(
        object_field(manifest, "modelContract")
    )
    data_contract = object_field(manifest, "dataContract")
    digests = resolved_semantic_digests(
        model_contract,
        string_field(data_contract, "dataContractSha256"),
        model_config,
    )
    if digests != object_field(manifest, "semanticDigests"):
        raise ValueError("worker semantic digests do not match modelContract")
    return model_contract


def _load_initialization(
    manifest: JsonObject,
    device: torch.device,
    model_contract: ModelContract,
) -> tuple[JsonObject, dict[str, object] | None]:
    initialization = object_field(manifest, "initialization")
    source = string_field(initialization, "source")
    if source == "random":
        return {"source": "random"}, None
    if source != "publishedModel":
        raise ValueError("fit initialization source is invalid")

    artifact = object_field(object_field(manifest, "model"), "parentCheckpoint")
    if string_field(initialization, "parentCheckpointSha256") != string_field(
        artifact,
        "checkpointSha256",
    ):
        raise ValueError("parent checkpoint digest differs from initialization")
    checkpoint = _load_model_checkpoint(
        validate_checkpoint_artifact(artifact),
        device,
        mismatch_message="parent checkpoint belongs to another model contract",
    )
    metadata = object_document(checkpoint["metadata"], "checkpoint metadata")
    semantic_digests = object_field(manifest, "semanticDigests")
    if (
        metadata.get("modelContract") != model_contract.to_document()
        or metadata.get("semanticDigests") != semantic_digests
    ):
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "parent checkpoint contract differs from the fit job",
        )
    _validate_initialization_digests(initialization, semantic_digests)
    return dict(initialization), checkpoint


def _restore_recovery(
    trainer: Trainer,
    recovery: JsonObject,
    device: torch.device,
    manifest: JsonObject,
) -> None:
    if any(
        recovery.get(field) != manifest.get(field)
        for field in (
            "jobId",
            "inputRevision",
            "jobConfigSha256",
            "semanticDigests",
            "manifestSha256",
        )
    ):
        raise WorkerExecutionError(
            "RECOVERY_CHECKPOINT_INCOMPATIBLE",
            "training recovery fences differ from the job",
        )
    artifact = object_field(recovery, "checkpoint")
    checkpoint_path = validate_checkpoint_artifact(artifact)
    try:
        payload = load_training_recovery(
            checkpoint_path,
            device,
            descriptor=recovery,
        )
        metadata = object_document(payload["metadata"], "checkpoint metadata")
        if (
            metadata.get("dataContract") != manifest.get("dataContract")
            or metadata.get("modelContract") != manifest.get("modelContract")
        ):
            raise ValueError("recovery semantic contracts differ")
        trainer.load_recovery_state_dict(
            object_document(payload["trainer_state"], "trainer state")
        )
    except Exception as exc:
        raise WorkerExecutionError(
            "RECOVERY_CHECKPOINT_INCOMPATIBLE",
            "training recovery checkpoint could not be restored",
        ) from exc


def _load_model_checkpoint(
    path: str,
    device: torch.device,
    *,
    mismatch_message: str,
) -> dict[str, object]:
    try:
        return load_checkpoint(path, device)
    except CheckpointFormatMismatch as exc:
        raise WorkerExecutionError("MODEL_SCHEMA_MISMATCH", mismatch_message) from exc
    except CheckpointCorrupt as exc:
        raise WorkerExecutionError(
            "MODEL_CORRUPT",
            "checkpoint semantic metadata is invalid",
        ) from exc


def _validate_initialization_digests(
    initialization: Mapping[str, object],
    semantic_digests: Mapping[str, object],
) -> None:
    pairs = (
        ("parentDataContractSha256", "dataContractSha256"),
        ("dataContractSha256", "dataContractSha256"),
        ("parentTargetContractSha256", "targetContractSha256"),
        ("targetContractSha256", "targetContractSha256"),
        ("parentObjectiveSha256", "objectiveSha256"),
        ("objectiveSha256", "objectiveSha256"),
        ("parentModelDefinitionSha256", "modelDefinitionSha256"),
        ("modelDefinitionSha256", "modelDefinitionSha256"),
    )
    if any(
        initialization.get(initialization_key)
        != semantic_digests.get(digest_key)
        for initialization_key, digest_key in pairs
    ):
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "published model initialization digests differ from the fit job",
        )


def _completed_manifest(
    manifest: JsonObject,
    input_stream: DurableInputStream,
) -> JsonObject:
    if input_stream.manifest_sha256 is None:
        raise ValueError("closed input manifest digest is unavailable")
    completed = dict(manifest)
    completed["inputRevision"] = input_stream.input_revision
    completed["manifestSha256"] = input_stream.manifest_sha256
    return completed


def _training_metrics(
    metrics: ObservedTrainingEpoch,
    epoch: int,
    monitor_payload: SelectionPayload,
    model_contract: ModelContract,
) -> JsonObject | None:
    try:
        document = epoch_telemetry_document(
            metrics,
            mode="fit-stream",
            frame=None,
            epoch=epoch + 1,
            selectionScore=monitor_payload["selection_score"],
            checkpointBest=monitor_payload["checkpoint_best"],
            shouldStop=monitor_payload["should_stop"],
            bestSelectionScore=monitor_payload["best_selection_score"],
        )
        if document is None:
            raise ValueError("training epoch telemetry is unavailable")
        result = object_document(json_safe(document), "committed fit metrics")
        return validate_training_metrics_for_model(result, model_contract)
    except Exception as exc:
        print(
            "training epoch telemetry disabled: " + type(exc).__name__,
            file=sys.stderr,
            flush=True,
        )
        return None


__all__ = ["execute_fit"]
