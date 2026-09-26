from __future__ import annotations

import os
from collections.abc import Iterator
from itertools import chain

import torch

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v19 import (
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    validate_document,
)
from app.contracts.worker.v19.config import ModelConfig, TrainConfig
from app.contracts.worker.v19.model_definition import resolved_semantic_digests
from app.worker.application.artifacts import (
    CommittedInputArtifacts,
    artifact_document,
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
from app.worker.data.arrow import iter_committed_source_arrow, write_arrow
from app.worker.data.tensors import validate_checkpoint_feature_dim
from app.worker.runtime.device import get_device
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import Trainer


def execute_predict(
    manifest: JsonObject,
    workspace: str,
    input_stream: DurableInputStream,
    emitter: WorkerEventEmitter,
) -> JsonObject:
    model_config = ModelConfig.from_manifest(
        object_field(manifest, "modelConfig")
    )
    model_contract = _validated_model_contract(manifest, model_config)
    model_document = object_field(manifest, "model")
    checkpoint_path = validate_checkpoint_artifact(
        object_field(model_document, "checkpoint")
    )
    device = get_device(
        string_field(object_field(manifest, "device"), "backend")
    )
    checkpoint = _load_checkpoint(checkpoint_path, device)
    metadata = object_document(checkpoint["metadata"], "checkpoint metadata")
    semantic_digests = object_field(manifest, "semanticDigests")
    if (
        metadata.get("modelContract") != model_contract.to_document()
        or metadata.get("semanticDigests") != semantic_digests
        or metadata.get("predictionDefinition")
        != model_contract.prediction_definition(model_config.seq_len)
    ):
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "prediction checkpoint contract differs from the job",
        )
    train_config = _training_config(metadata)
    prediction_column = string_field(manifest, "predictionColumn")
    source_encoding = object_field(manifest, "sourceEncoding")
    committed_inputs = CommittedInputArtifacts()
    model: torch.nn.Module | None = None
    trainer: Trainer | None = None
    artifacts: list[JsonValue] = []
    for item in input_stream.items():
        if string_field(item, "schemaId") != PREDICT_INPUT_SCHEMA_ID:
            raise ValueError("prediction input schemaId is invalid")
        feature_slices = iter_committed_source_arrow(
            committed_inputs.path(item),
            expected_rows=integer_field(item, "logicalRows"),
            expected_chunks=integer_field(item, "chunks"),
            expected_native_rows=integer_list(item.get("nativeRows"), "nativeRows"),
            source_encoding=source_encoding,
            seq_len=model_config.seq_len,
            feature_dim=model_config.feature_dim,
            target_contract=model_contract.target_contract,
        )
        output_path = os.path.join(
            workspace,
            "outputs",
            f"{integer_field(item, 'ordinal')}.arrow",
        )
        first = next(feature_slices, None)
        if first is None:
            predictions = torch.empty(
                (0, model_contract.target_width),
                dtype=torch.float32,
            )
        else:
            validate_checkpoint_feature_dim(first, model_config.feature_dim)
            if model is None:
                model = build_model(
                    model_config,
                    first,
                    None,
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
                        semantic_digests,
                        "modelDefinitionSha256",
                    ),
                    initialization=object_field(metadata, "initialization"),
                )
                trainer.load_payload(checkpoint)
            if trainer is None:
                raise AssertionError("prediction trainer was not initialized")
            predictions = _predict_slices(
                trainer,
                chain((first,), feature_slices),
            )
        write_arrow(
            output_path,
            predictions,
            prediction_column,
            model_contract.target_contract,
            expected_rows=integer_field(item, "logicalRows"),
        )
        artifacts.append({
            "schemaId": PREDICTION_OUTPUT_SCHEMA_ID,
            "ordinal": integer_field(item, "ordinal"),
            "commitRevision": integer_field(item, "commitRevision"),
            "dataContractSha256": string_field(item, "dataContractSha256"),
            "modelDefinitionSha256": string_field(
                semantic_digests,
                "modelDefinitionSha256",
            ),
            "rows": integer_field(item, "logicalRows"),
            "artifact": artifact_document(output_path),
        })
        emitter.progress({
            "ordinal": integer_field(item, "ordinal"),
            "rows": integer_field(item, "logicalRows"),
        })
    result = result_identity(manifest)
    result["inputRevision"] = input_stream.input_revision
    result["manifestSha256"] = input_stream.manifest_sha256
    result["artifacts"] = artifacts
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
    expected = resolved_semantic_digests(
        model_contract,
        string_field(data_contract, "dataContractSha256"),
        model_config,
    )
    if expected != object_field(manifest, "semanticDigests"):
        raise ValueError("worker semantic digests do not match modelContract")
    return model_contract


def _training_config(metadata: JsonObject) -> TrainConfig:
    manifest = dict(object_field(metadata, "trainingConfig"))
    manifest["diagnostics"] = object_field(metadata, "diagnostics")
    config = TrainConfig.from_dict(manifest)
    if config is None:
        raise WorkerExecutionError(
            "MODEL_CORRUPT",
            "prediction checkpoint training configuration is unavailable",
        )
    return config


def _load_checkpoint(
    path: str,
    device: torch.device,
) -> dict[str, object]:
    try:
        return load_checkpoint(path, device)
    except CheckpointFormatMismatch as exc:
        raise WorkerExecutionError(
            "MODEL_SCHEMA_MISMATCH",
            "prediction checkpoint belongs to another model contract",
        ) from exc
    except CheckpointCorrupt as exc:
        raise WorkerExecutionError(
            "MODEL_CORRUPT",
            "prediction checkpoint semantic metadata is invalid",
        ) from exc


def _predict_slices(
    trainer: Trainer,
    slices: Iterator[torch.Tensor],
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    pending: torch.Tensor | None = None
    for features in slices:
        if pending is not None:
            needed = trainer.batch_size - pending.size(0)
            combined = torch.cat((pending, features[:needed]), dim=0)
            if combined.size(0) == trainer.batch_size:
                predictions.append(trainer.predict(combined).cpu())
                features = features[needed:]
                pending = None
            else:
                pending = combined
                continue
        full_rows = (features.size(0) // trainer.batch_size) * trainer.batch_size
        for offset in range(0, full_rows, trainer.batch_size):
            predictions.append(
                trainer.predict(features[offset:offset + trainer.batch_size]).cpu()
            )
        if full_rows < features.size(0):
            pending = features[full_rows:]
    if pending is not None:
        predictions.append(trainer.predict(pending).cpu())
    if not predictions:
        raise ValueError("prediction input produced no logical rows")
    return torch.cat(predictions, dim=0)


__all__ = ["execute_predict"]
