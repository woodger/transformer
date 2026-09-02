from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from dataclasses import replace

import torch

from app.contracts.json_types import JsonObject
from app.contracts.worker.v10 import FIT_INPUT_SCHEMA_ID, validate_document
from app.contracts.worker.v10.config import ModelConfig, TrainConfig
from app.contracts.worker.v10.diagnostics import DiagnosticsConfig
from app.contracts.worker.v10.objective import objective_from_ml_contract
from app.worker.application.artifacts import (
    CommittedInputArtifacts,
    artifact_document,
    boolean_value,
    checkpoint_metadata,
    json_safe,
    result_identity,
    validate_artifact,
)
from app.worker.application.documents import (
    integer_field,
    object_document,
    object_field,
    optional_string_field,
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
from app.worker.data.arrow import read_committed_fit_arrow
from app.worker.data.tensors import (
    TrainingBatch,
    reshape_source,
    validate_feature_dim,
    validate_target_dim,
)
from app.worker.runtime.device import get_device
from app.worker.runtime.reproducibility import configure_reproducibility
from app.worker.telemetry import epoch_telemetry_document
from app.worker.telemetry.epoch import ObservedTrainingEpoch
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import SelectionPayload


def execute_fit(
    manifest: JsonObject,
    workspace: str,
    input_stream: DurableInputStream,
    emitter: WorkerEventEmitter,
) -> JsonObject:
    model_document = object_field(manifest, "model")
    model_config = ModelConfig.from_dict(
        object_field(model_document, "config")
    )
    train_config = TrainConfig.from_dict(object_field(manifest, "training"))
    if model_config is None or train_config is None:
        raise ValueError("fit configuration is unavailable")
    objective = objective_from_ml_contract(object_field(manifest, "mlContract"))
    diagnostics = DiagnosticsConfig.from_document(
        object_field(manifest, "diagnostics")
    )
    train_config = replace(train_config, diagnostics=diagnostics)
    if model_config.out_dim != objective.target_width:
        raise ValueError("fit model shape differs from objective targets")
    configure_reproducibility(train_config.seed, train_config.deterministic)
    device = get_device(string_field(object_field(manifest, "device"), "kind"))
    data_contract = object_field(manifest, "dataContract")
    ml_contract = object_field(manifest, "mlContract")
    expected_feature_dim = integer_field(data_contract, "featureDim")
    expected_target_dim = objective.target_width
    committed_inputs = CommittedInputArtifacts()

    def read_payload(item: JsonObject) -> TrainingBatch:
        if string_field(item, "schemaId") != FIT_INPUT_SCHEMA_ID:
            raise ValueError("fit input schemaId is invalid")
        path = committed_inputs.path(item)
        batch = read_committed_fit_arrow(
            path,
            expected_rows=integer_field(item, "rows"),
            source_width=model_config.seq_len * expected_feature_dim,
            targets=objective.targets,
        )
        batch = TrainingBatch(
            features=reshape_source(batch.features, model_config.seq_len),
            targets=batch.targets,
        )
        validate_feature_dim(batch.features, expected_feature_dim)
        validate_target_dim(batch.targets, expected_target_dim)
        return batch

    stream = iter(input_stream.items())
    first = None
    for item in stream:
        batch = read_payload(item)
        if batch.features.size(0) == 0:
            continue
        first = batch
        break
    if first is None:
        raise ValueError("fit requires at least one non-empty input")

    actual_config = replace(model_config, feature_dim=expected_feature_dim)
    initialization, parent_checkpoint = _load_initialization(
        manifest,
        device,
        actual_config,
        data_contract,
        ml_contract,
    )
    model = build_model(
        actual_config,
        first.features,
        first.targets,
        device,
        objective,
    )
    trainer = build_trainer(
        train_config,
        model,
        device,
        actual_config,
        data_contract=data_contract,
        objective=objective,
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
    if recovery is not None and recovery.get("checkpoint") is not None:
        if (
            not input_stream.closed
            or optional_string_field(recovery, "manifestSha256") is None
        ):
            raise ValueError(
                "recovery checkpoint requires closed immutable input"
            )
        checkpoint_path = validate_artifact(
            object_field(recovery, "checkpoint")
        )
        try:
            payload = load_training_recovery(
                checkpoint_path,
                device,
                expected_config_hash=string_field(recovery, "configSha256"),
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
                or payload["model_config"] != trainer.model_config.to_dict()
            ):
                raise ValueError("recovery model configuration differs")
            if payload["train_config"] != train_config.to_dict():
                raise ValueError("recovery training configuration differs")
            if payload["data_contract"] != data_contract:
                raise ValueError("recovery data contract differs")
            trainer.load_recovery_state_dict(
                object_document(payload["trainer_state"], "trainer state")
            )
        except Exception as exc:
            raise WorkerExecutionError(
                "RECOVERY_CHECKPOINT_INCOMPATIBLE",
                "training recovery checkpoint could not be restored",
            ) from exc

    def first_epoch_payloads() -> Iterator[TrainingBatch]:
        yield first
        for item in stream:
            batch = read_payload(item)
            if batch.features.size(0) != 0:
                yield batch

    def closed_payloads() -> Iterator[TrainingBatch]:
        if not input_stream.closed:
            raise ValueError("complete input is unavailable for replay")
        for item in input_stream.inputs:
            batch = read_payload(item)
            if batch.features.size(0) != 0:
                yield batch

    def on_epoch_committed(
        epoch: int,
        metrics: ObservedTrainingEpoch,
        monitor_payload: SelectionPayload,
        _training_complete: bool,
    ) -> None:
        if recovery is None:
            return
        manifest_sha256 = input_stream.manifest_sha256
        if manifest_sha256 is None:
            raise ValueError("recovery checkpoint cannot precede input EOF")
        generation = trainer.state.global_epoch
        checkpoint_path = os.path.join(
            workspace,
            "checkpoints",
            f"{generation}.pth",
        )
        serialization_started = time.monotonic()
        event = save_training_recovery(
            checkpoint_path,
            trainer,
            generation=generation,
            config_hash=string_field(recovery, "configSha256"),
            manifest_hash=manifest_sha256,
        )
        checkpoint_serialization_ms = (
            time.monotonic() - serialization_started
        ) * 1000.0
        metrics_payload = {
            "mode": "fit-stream",
            "frame": None,
            "epoch": epoch + 1,
            "selection_score": monitor_payload["selection_score"],
            "checkpoint_best": monitor_payload["checkpoint_best"],
            "should_stop": monitor_payload["should_stop"],
            "best_selection_score": monitor_payload["best_selection_score"],
        }
        committed_metrics: JsonObject | None = None
        try:
            telemetry_document = epoch_telemetry_document(
                metrics,
                **metrics_payload,
            )
            if telemetry_document is None:
                raise ValueError("training epoch telemetry is unavailable")
            committed_metrics = object_document(
                json_safe(telemetry_document),
                "committed fit metrics",
            )
            validate_document(committed_metrics, "training-metrics")
        except Exception as exc:
            print(
                "training epoch telemetry disabled: " + type(exc).__name__,
                file=sys.stderr,
                flush=True,
            )
        checkpoint_event: JsonObject = {
            "generation": integer_field(event, "generation"),
            "completedEpochs": integer_field(event, "completed_epochs"),
            "globalStep": integer_field(event, "global_step"),
            "progress": {
                "epoch": epoch + 1,
                "step": metrics.step,
                "loss": metrics.loss,
            },
            "trainingComplete": boolean_value(
                event.get("training_complete"),
                "training_complete",
            ),
            "artifact": artifact_document(checkpoint_path),
            "checkpointSerializationMs": checkpoint_serialization_ms,
        }
        if committed_metrics is not None:
            checkpoint_event["metrics"] = committed_metrics
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

    checkpoint_path = os.path.join(workspace, "checkpoint.pth")
    serialization_started = time.monotonic()
    trainer.save(checkpoint_path)
    checkpoint_serialization_ms = (
        time.monotonic() - serialization_started
    ) * 1000.0
    if not input_stream.closed:
        raise ValueError("fit result requires a closed immutable input")
    result = result_identity(manifest)
    result_fields: JsonObject = {
        "inputRevision": input_stream.input_revision,
        "manifestSha256": input_stream.manifest_sha256,
        "artifacts": [],
        "checkpoint": artifact_document(checkpoint_path),
        "checkpointMetadata": checkpoint_metadata(trainer, data_contract),
        "checkpointSerializationMs": checkpoint_serialization_ms,
    }
    result.update(result_fields)
    validate_document(result, "result-manifest")
    return result


def _load_initialization(
    manifest: JsonObject,
    device: torch.device,
    model_config: ModelConfig,
    data_contract: JsonObject,
    ml_contract: JsonObject,
) -> tuple[JsonObject, dict[str, object] | None]:
    initialization = object_field(manifest, "initialization")
    kind = string_field(initialization, "kind")
    if kind == "random":
        return {"kind": "random"}, None

    checkpoint_document = object_field(initialization, "checkpoint")
    parent_sha256 = string_field(
        initialization,
        "parentCheckpointSha256",
    )
    if parent_sha256 != string_field(checkpoint_document, "sha256"):
        raise ValueError(
            "parent checkpoint digest differs from initialization"
        )
    checkpoint_path = validate_artifact(checkpoint_document)
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
    except CheckpointFormatMismatch as exc:
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "parent checkpoint belongs to another ML contract",
        ) from exc
    except CheckpointCorrupt as exc:
        raise WorkerExecutionError(
            "MODEL_CORRUPT",
            "parent checkpoint semantic metadata is invalid",
        ) from exc
    if (
        checkpoint["model_config"] != model_config.to_dict()
        or checkpoint["ml_contract"] != ml_contract
    ):
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "parent checkpoint contract differs from the fit job",
        )
    parent_data_contract = checkpoint.get("data_contract")
    if kind == "publishedModel":
        if parent_data_contract != data_contract:
            raise WorkerExecutionError(
                "MODEL_SCHEMA_MISMATCH",
                "parent checkpoint data contract differs from the fit job",
            )
        lineage: JsonObject = {
            "kind": "publishedModel",
            "parentModelRef": string_field(
                initialization,
                "parentModelRef",
            ),
            "parentCheckpointSha256": parent_sha256,
        }
        return lineage, checkpoint

    parent_contract = object_document(
        parent_data_contract,
        "parent checkpoint data contract",
    )
    parent_data_contract_sha256 = string_field(
        initialization,
        "parentDataContractSha256",
    )
    current_data_contract_sha256 = string_field(
        initialization,
        "dataContractSha256",
    )
    if (
        string_field(parent_contract, "dataContractSha256")
        != parent_data_contract_sha256
    ):
        raise WorkerExecutionError(
            "MODEL_CORRUPT",
            "parent checkpoint data contract digest differs from initialization",
        )
    if (
        string_field(data_contract, "dataContractSha256")
        != current_data_contract_sha256
    ):
        raise ValueError(
            "fit data contract digest differs from initialization"
        )
    if not _transfer_data_contracts_compatible(
        parent_contract,
        data_contract,
    ):
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "parent model data contract is not compatible with transfer fit",
        )
    return (
        {
            "kind": "publishedModelTransfer",
            "parentModelRef": string_field(
                initialization,
                "parentModelRef",
            ),
            "parentCheckpointSha256": parent_sha256,
            "parentDataContractSha256": parent_data_contract_sha256,
            "dataContractSha256": current_data_contract_sha256,
        },
        checkpoint,
    )


def _transfer_data_contracts_compatible(
    parent: JsonObject,
    current: JsonObject,
) -> bool:
    fields = (
        "id",
        "version",
        "profile",
        "seqLen",
        "featureDim",
        "targetSchemaId",
    )
    return all(parent.get(field) == current.get(field) for field in fields)


__all__ = ["execute_fit"]
