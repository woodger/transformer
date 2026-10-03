import json

from app.project import PROJECT_ROOT

CHECKPOINT_SCHEMAS = (
    "checkpoint-artifact",
    "checkpoint-metadata",
    "recovery-metadata",
    "resolved-initialization",
)
WORKER_SCHEMAS = (
    "arrow-manifest",
    "capabilities",
    "command-manifest",
    "control-message",
    "event",
    "prediction-manifest",
    "recovery-descriptor",
    "result-manifest",
    "target-head-diagnostics-artifact",
    "training-metrics",
)


def internal_contract_documents(
    *,
    checkpoint_version,
    worker_version,
    semantic_version,
    target_head_version,
):
    contracts = PROJECT_ROOT / "app/contracts"
    semantic_root = contracts / "semantic" / f"v{semantic_version}" / "fixtures"
    fixture = json.loads(
        (semantic_root / "positive-class-weighted-binary-w28.json").read_text(),
    )
    report = json.loads((
        contracts / "target_head_diagnostics" / f"v{target_head_version}"
        / "fixtures/report.result.available.encoder-learning-no-finite-gradient.json"
    ).read_text())
    job_id = report["producingRunId"]
    attempt_id = "22222222-2222-4222-8222-222222222222"
    model_contract = fixture["modelContract"]
    target = model_contract["targetContract"]["slots"][0]["identity"]
    digests = {
        "dataContractSha256": "a" * 64,
        "targetContractSha256": fixture["expected"]["targetContractSha256"],
        "objectiveSha256": fixture["expected"]["objectiveSha256"],
        "modelDefinitionSha256": report["modelDefinitionSha256"],
    }
    data_contract = {
        "dataContractSha256": "a" * 64,
        "seqLen": 2,
        "featureDim": 8,
    }
    model_config = {"seqLen": 2, "featureDim": 8, **model_contract["modelTuning"]}
    training = {
        "lr": 0.001,
        "batchSize": 2,
        "epochs": 1,
        "useAmp": False,
        "weightDecay": 0.0,
        "selection": None,
        "seed": 17,
        "deterministic": True,
    }
    diagnostics = {
        "schemaVersion": 3,
        "gradientInteractions": None,
        "targetHead": "fullCommittedArtifact",
        "encoderLayerDiagnostics": "directComponentPerBatch",
    }
    progress = {"completedEpochs": 1, "globalStep": 1, "trainingComplete": True}
    initialization = {"source": "random"}
    checkpoint_artifact = {
        "format": f"transformer-checkpoint-v{checkpoint_version}",
        "byteCount": 1,
        "checkpointSha256": "b" * 64,
    }
    metadata = {
        "format": checkpoint_artifact["format"],
        "serviceVersion": "0.1.21",
        "generation": 1,
        "jobId": job_id,
        "dataContract": data_contract,
        "modelContract": model_contract,
        "predictionDefinition": {
            "seqLen": 2,
            "outputWidth": 1,
            "targets": [{
                "identity": target,
                "publicPredictionTransformation": "Sigmoid",
                "positiveClassWeight": 28.0,
            }],
        },
        "modelConfig": model_config,
        "semanticDigests": digests,
        "trainingConfig": training,
        "diagnostics": diagnostics,
        "selection": {
            "enabled": False,
            "modelDefinitionSha256": digests["modelDefinitionSha256"],
            "bestSelectionScore": None,
            "bestEpoch": None,
            "source": "last_epoch",
        },
        "initialization": initialization,
        "jobConfigSha256": "c" * 64,
        "manifestSha256": "d" * 64,
        "progress": progress,
    }
    recovery = {
        "format": f"transformer-recovery-v{checkpoint_version}",
        "jobId": job_id,
        "generation": 1,
        "inputRevision": 1,
        "jobConfigSha256": metadata["jobConfigSha256"],
        "semanticDigests": digests,
        "manifestSha256": metadata["manifestSha256"],
        "checkpoint": checkpoint_artifact,
        "progress": progress,
    }
    artifact = {"path": "input.arrow", "byteCount": 1, "sha256": "e" * 64}
    input_manifest = {
        "schemaId": "transformer.indexed-feature-blocks.fit.v1",
        "ordinal": 0,
        "commitRevision": 1,
        "dataContractSha256": data_contract["dataContractSha256"],
        "chunks": 1,
        "logicalRows": 1,
        "nativeRows": [2],
        "firstRangeOrdinal": 0,
        "firstExampleOffset": 0,
        "lastRangeOrdinal": 0,
        "nextExampleOffset": 1,
        "batches": 1,
        "artifact": artifact,
    }
    envelope = {
        "contract": "transformer-worker",
        "protocolVersion": worker_version,
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
    }
    command = {
        **envelope,
        "operation": "fit",
        "device": {"backend": "cpu"},
        "inputs": [input_manifest],
        "inputRevision": 1,
        "inputClosed": True,
        "manifestSha256": metadata["manifestSha256"],
        "workspace": {"root": "workspace"},
        "model": {"label": "binary"},
        "sourceEncoding": {
            "featureBlocks": [{"windowRows": 2, "nativeRowWidth": 4}],
        },
        "dataContract": data_contract,
        "modelContract": model_contract,
        "modelConfig": model_config,
        "semanticDigests": digests,
        "jobConfigSha256": metadata["jobConfigSha256"],
        "training": training,
        "diagnostics": diagnostics,
        "initialization": initialization,
        "recovery": None,
    }
    target_head = {
        "format": f"transformer-target-head-diagnostics-v{target_head_version}",
        "jobId": job_id,
        "attempt": 1,
        "attemptId": attempt_id,
        "inputRevision": 1,
        "manifestSha256": metadata["manifestSha256"],
        "modelDefinitionSha256": digests["modelDefinitionSha256"],
        "jobConfigSha256": metadata["jobConfigSha256"],
        "sampleIdentity": report["sampleIdentity"],
        "sampleRowCount": report["sampleRowCount"],
        "layout": report["layout"],
        "coverage": report["coverage"],
        "epochs": report["epochPage"]["items"],
    }
    component = model_contract["objective"]["directComponents"][0]
    metrics = {
        "mode": "fit-stream",
        "frame": None,
        "epoch": 1,
        "step": 1,
        "rows": 1,
        "batches": 1,
        "lr": training["lr"],
        "loss": 0.5,
        "directLosses": [{
            "componentIdentity": component["identity"],
            "operator": component["operator"],
            "targetIdentity": target,
            "targetIndex": 0,
            "value": 0.5,
        }],
        "auxiliaryLosses": [],
        "targetMetrics": [{
            "targetIdentity": target,
            "targetIndex": 0,
            "mae": 0.1,
            "rmse": 0.1,
        }],
        "gradientInteractions": None,
        "selectionScore": None,
        "trainingBatchesCompleted": 1,
        "optimizerUpdatesApplied": 1,
        "optimizerUpdatesSkipped": 0,
        "ampOverflowBatches": 0,
        "finiteGradientBatches": 1,
        "nonFiniteGradientBatches": 0,
        "preClipGradientNormMean": 1.0,
        "preClipGradientNormMax": 1.0,
        "preClipGradientNormP95": 1.0,
        "nanRatio": 0.0,
        "maskedTokenRatio": 0.0,
        "completeTokenRatio": 1.0,
        "partialTokenRatio": 0.0,
        "emptyTokenRatio": 0.0,
        "inputPipelineMs": 1.0,
        "missingStatsMs": 1.0,
        "hostToDeviceMs": 1.0,
        "trainStepMs": 1.0,
        "elapsedMs": 4.0,
        "checkpointBest": False,
        "shouldStop": False,
        "bestSelectionScore": None,
    }

    return {
        "checkpoint": {
            "checkpoint-artifact": checkpoint_artifact,
            "checkpoint-metadata": metadata,
            "recovery-metadata": recovery,
            "resolved-initialization": initialization,
        },
        "worker": {
            "arrow-manifest": input_manifest,
            "capabilities": {
                "contract": envelope["contract"],
                "protocolVersion": worker_version,
                "checkpointFormat": checkpoint_artifact["format"],
                "recoveryFormat": recovery["format"],
                "schemaIds": {
                    "fitInput": input_manifest["schemaId"],
                    "predictInput": "transformer.indexed-feature-blocks.predict.v1",
                    "predictionOutput": "transformer.prediction.target-aligned.v3",
                },
                "semantic": json.loads((semantic_root / "language-capabilities.json").read_text()),
                "torchVersion": "2.14.0",
                "cudaRuntimeVersion": None,
                "devices": [{"backend": "cpu", "opaqueId": "cpu", "name": "CPU"}],
            },
            "command-manifest": command,
            "control-message": {**envelope, "sequence": 1, "type": "cancel", "payload": {}},
            "event": {
                **envelope,
                "sequence": 1,
                "type": "ready",
                "payload": {"pid": 1, "nextOrdinal": 0, "inputRevision": 1},
            },
            "prediction-manifest": {
                "schemaId": "transformer.prediction.target-aligned.v3",
                "ordinal": 0,
                "commitRevision": 1,
                "dataContractSha256": data_contract["dataContractSha256"],
                "modelDefinitionSha256": digests["modelDefinitionSha256"],
                "rows": 1,
                "artifact": {**artifact, "path": "prediction.arrow"},
            },
            "recovery-descriptor": {
                **recovery,
                "checkpoint": {**checkpoint_artifact, "path": "recovery.pth"},
            },
            "result-manifest": {
                **envelope,
                "operation": "fit",
                "inputRevision": 1,
                "manifestSha256": metadata["manifestSha256"],
                "jobConfigSha256": metadata["jobConfigSha256"],
                "semanticDigests": digests,
                "artifacts": [],
                "checkpoint": {**checkpoint_artifact, "path": "checkpoint.pth"},
                "checkpointMetadata": metadata,
                "checkpointSerializationMs": 1.0,
                "targetHeadDiagnostics": target_head,
            },
            "target-head-diagnostics-artifact": target_head,
            "training-metrics": metrics,
        },
    }
