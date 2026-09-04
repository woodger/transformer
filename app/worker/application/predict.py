from __future__ import annotations

import os
from collections.abc import Iterator
from itertools import chain

import torch

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v11 import (
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    validate_document,
)
from app.contracts.worker.v11.config import ModelConfig, TrainConfig
from app.contracts.worker.v11.objective import objective_from_ml_contract
from app.worker.application.artifacts import (
    CommittedInputArtifacts,
    artifact_document,
    result_identity,
    validate_artifact,
)
from app.worker.application.documents import (
    integer_field,
    integer_list,
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
from app.worker.data.tensors import (
    validate_checkpoint_feature_dim,
)
from app.worker.runtime.device import get_device
from app.worker.training.factory import build_model, build_trainer
from app.worker.training.trainer import Trainer


def execute_predict(
    manifest: JsonObject,
    workspace: str,
    input_stream: DurableInputStream,
    emitter: WorkerEventEmitter,
) -> JsonObject:
    model_document = object_field(manifest, "model")
    checkpoint_path = validate_artifact(
        object_field(model_document, "checkpoint")
    )
    device = get_device(string_field(object_field(manifest, "device"), "kind"))
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
    try:
        objective = objective_from_ml_contract(checkpoint["ml_contract"])
    except (TypeError, ValueError) as exc:
        raise WorkerExecutionError(
            "MODEL_CORRUPT",
            "prediction checkpoint objective is invalid",
        ) from exc
    if (
        checkpoint["data_contract"] != object_field(manifest, "dataContract")
        or checkpoint["ml_contract"] != object_field(manifest, "mlContract")
        or model_config.out_dim != objective.target_width
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
    source_encoding = object_field(manifest, "sourceEncoding")
    committed_inputs = CommittedInputArtifacts()
    for item in input_stream.items():
        if string_field(item, "schemaId") != PREDICT_INPUT_SCHEMA_ID:
            raise ValueError("prediction input schemaId is invalid")
        input_path = committed_inputs.path(item)
        feature_slices = iter_committed_source_arrow(
            input_path,
            expected_rows=integer_field(item, "logicalRows"),
            expected_chunks=integer_field(item, "chunks"),
            expected_native_rows=integer_list(
                item.get("nativeRows"),
                "nativeRows",
            ),
            source_encoding=source_encoding,
            seq_len=model_config.seq_len,
            feature_dim=model_config.feature_dim,
        )
        output_path = os.path.join(
            workspace,
            "outputs",
            f"{integer_field(item, 'ordinal')}.arrow",
        )
        first = next(feature_slices, None)
        if first is None:
            predictions = torch.empty(
                (0, objective.target_width),
                dtype=torch.float32,
            )
        else:
            validate_checkpoint_feature_dim(
                first,
                model_config.feature_dim,
            )
            if model is None:
                model = build_model(
                    model_config,
                    first,
                    None,
                    device,
                    objective,
                )
                trainer = build_trainer(
                    train_config,
                    model,
                    device,
                    model_config,
                    data_contract=data_contract,
                    objective=objective,
                )
                if checkpoint is None:
                    raise AssertionError(
                        "prediction checkpoint was already consumed"
                    )
                trainer.load_payload(checkpoint)
                checkpoint = None
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
            expected_rows=integer_field(item, "logicalRows"),
            targets=objective.targets,
        )
        artifacts.append({
            "schemaId": PREDICTION_OUTPUT_SCHEMA_ID,
            "ordinal": integer_field(item, "ordinal"),
            "commitRevision": integer_field(item, "commitRevision"),
            "dataContractSha256": string_field(
                item,
                "dataContractSha256",
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
