from __future__ import annotations

import os

import torch

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v5 import (
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    validate_document,
)
from app.contracts.worker.v5.config import ModelConfig, TrainConfig
from app.contracts.worker.v5.objective import ml_contract
from app.worker.application.artifacts import (
    CommittedInputArtifacts,
    artifact_document,
    result_identity,
    validate_artifact,
)
from app.worker.application.documents import integer_field, object_field, string_field
from app.worker.application.errors import WorkerExecutionError
from app.worker.application.events import WorkerEventEmitter
from app.worker.application.inputs import DurableInputStream
from app.worker.checkpoints.model import (
    CheckpointCorrupt,
    CheckpointFormatMismatch,
    load_checkpoint,
)
from app.worker.data.arrow import read_committed_source_arrow, write_arrow
from app.worker.data.tensors import (
    reshape_source,
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
    committed_inputs = CommittedInputArtifacts()
    for item in input_stream.items():
        if string_field(item, "schemaId") != PREDICT_INPUT_SCHEMA_ID:
            raise ValueError("prediction input schemaId is invalid")
        input_path = committed_inputs.path(item)
        features_cpu = read_committed_source_arrow(
            input_path,
            expected_rows=integer_field(item, "rows"),
            source_width=model_config.seq_len * model_config.feature_dim,
        )
        output_path = os.path.join(
            workspace,
            "outputs",
            f"{integer_field(item, 'ordinal')}.arrow",
        )
        if features_cpu.size(0) == 0:
            predictions = torch.empty((0, 6), dtype=torch.float32)
        else:
            features_cpu = reshape_source(features_cpu, model_config.seq_len)
            validate_checkpoint_feature_dim(
                features_cpu,
                model_config.feature_dim,
            )
            if model is None:
                model = build_model(model_config, features_cpu, None, device)
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
            predictions = trainer.predict(features_cpu)
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
            "artifact": artifact_document(output_path),
        })
        emitter.progress({
            "ordinal": integer_field(item, "ordinal"),
            "rows": integer_field(item, "rows"),
        })
    result = result_identity(manifest)
    result["inputRevision"] = input_stream.input_revision
    result["manifestSha256"] = input_stream.manifest_sha256
    result["artifacts"] = artifacts
    validate_document(result, "result-manifest")
    return result


__all__ = ["execute_predict"]
