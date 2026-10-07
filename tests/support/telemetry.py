from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.contracts.metrics.fit_run.v12 import build_run_summary
from app.contracts.metrics.v12 import (
    build_training_record,
)
from app.contracts.semantic.v6 import ModelContract
from app.contracts.worker.v21 import CHECKPOINT_FORMAT
from app.contracts.worker.v21.config import TrainConfig
from app.contracts.worker.v21.model_config import ModelConfig
from app.contracts.worker.v21.model_definition import resolved_semantic_digests
from tests.fixture_documents import semantic_fixture_document

DATA_CONTRACT_SHA256 = "d" * 64
MODEL_CONTRACT = ModelContract.from_document(
    semantic_fixture_document("single-regression")["modelContract"],
)
SEMANTIC_DIGESTS = resolved_semantic_digests(
    MODEL_CONTRACT,
    DATA_CONTRACT_SHA256,
    ModelConfig.from_tuning(
        MODEL_CONTRACT.model_tuning,
        seq_len=10,
        feature_dim=64848,
    ),
)
TARGET = MODEL_CONTRACT.target_identities[0]
DIRECT_COMPONENT = MODEL_CONTRACT.direct_components[0]


@dataclass(frozen=True, slots=True)
class TestMetricsArtifact:
    relative_path: str
    byte_count: int
    sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class TestRunSummaryArtifact:
    relative_path: str
    byte_count: int
    sha256: str


def train_config():
    return TrainConfig(
        batch_size=2,
        epochs=2,
        use_amp=False,
        deterministic=True,
        seed=17,
    )


def create_test_metrics_artifact(
    spool,
    *,
    model_ref,
    job_id,
    attempt_id,
    attempt,
):
    metrics = {
        "mode": "fit-stream",
        "frame": None,
        "epoch": 1,
        "step": 1,
        "rows": 1,
        "batches": 1,
        "lr": 0.001,
        "loss": 0.5,
        "directLosses": [
            {
                "componentIdentity": DIRECT_COMPONENT["identity"],
                "operator": DIRECT_COMPONENT["operator"],
                "targetIdentity": TARGET,
                "targetIndex": 0,
                "value": 0.1,
            }
        ],
        "auxiliaryLosses": [],
        "targetMetrics": [
            {
                "targetIdentity": TARGET,
                "targetIndex": 0,
                "mae": 0.1,
                "rmse": 0.1,
            }
        ],
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
    record = build_training_record(
        metrics,
        recorded_at=1.0,
        job_id=job_id,
        attempt_id=attempt_id,
        attempt=attempt,
        model_ref=model_ref,
        semantic_digests=SEMANTIC_DIGESTS,
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.10",
        git_commit="0" * 40,
        targets=MODEL_CONTRACT.target_identities,
    )
    payload = (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path = spool.telemetry_metrics_path(job_id)
    spool.atomic_write_bytes(path, payload)
    return TestMetricsArtifact(
        relative_path=spool.telemetry_relative_path(path),
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        row_count=1,
    )


def create_test_run_summary_artifact(
    spool,
    *,
    model_ref,
    job_id,
    attempt_id,
    attempt,
):
    summary = build_run_summary(
        recorded_at=10.0,
        job_id=job_id,
        attempt_id=attempt_id,
        attempt=attempt,
        model_ref=model_ref,
        semantic_digests=SEMANTIC_DIGESTS,
        job_config_sha256="e" * 64,
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.10",
        git_commit="0" * 40,
        targets=MODEL_CONTRACT.target_identities,
        initialization={"source": "random"},
        milestones={
            "createdAt": "1970-01-01T00:00:01.000Z",
            "firstInputCommittedAt": "1970-01-01T00:00:02.000Z",
            "inputClosedAt": "1970-01-01T00:00:03.000Z",
            "workerCompletedAt": "1970-01-01T00:00:09.000Z",
            "publishedAt": "1970-01-01T00:00:10.000Z",
        },
        durations={
            "firstInputWaitMs": 1000.0,
            "eofWaitMs": 1000.0,
            "queueWaitMs": 1000.0,
            "workerStartupMs": 1000.0,
            "trainingMs": 4000.0,
            "checkpointSerializationMs": 1.0,
            "checkpointPublicationMs": 1.0,
            "modelPublicationMs": 1000.0,
            "remoteFitMs": 9000.0,
        },
        counts={
            "attempts": attempt,
            "recoveries": max(0, attempt - 1),
            "inputPayloads": 1,
            "inputChunks": 1,
            "logicalRows": 1,
            "nativeRows": [1],
            "inputBytes": 1,
        },
    )
    payload = json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path = spool.telemetry_run_summary_path(job_id)
    spool.atomic_write_bytes(path, payload)
    return TestRunSummaryArtifact(
        relative_path=spool.telemetry_relative_path(path),
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
