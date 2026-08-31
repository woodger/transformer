from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.contracts.metrics.fit_run.v3 import build_run_summary
from app.contracts.metrics.v4 import (
    build_training_record,
)
from app.contracts.worker.v8.config import TrainConfig
from app.contracts.worker.v8.objective import (
    CHECKPOINT_FORMAT,
    default_objective,
    objective_config_sha256,
)

DATA_CONTRACT_SHA256 = "d" * 64


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
                "target": {"index": index, "name": name},
                "value": 0.1,
            }
            for index, name in enumerate((
                "MeanReturn",
                "SigmaReturn",
                "ProbTP",
                "ProbSL",
                "VolatilityNext",
                "HittingProbTP",
            ))
        ],
        "auxiliaryLosses": [
            {"operator": "GaussianNLL", "value": 0.0},
            {"operator": "RiskAdjustedExpectedValue", "value": 0.0},
        ],
        "targetMetrics": [
            {
                "target": {"index": index, "name": name},
                "mae": 0.1,
                "rmse": 0.1,
            }
            for index, name in enumerate((
                "MeanReturn",
                "SigmaReturn",
                "ProbTP",
                "ProbSL",
                "VolatilityNext",
                "HittingProbTP",
            ))
        ],
        "gradientInteractions": None,
        "selection_score": None,
        "trainingBatchesCompleted": 1,
        "optimizerUpdatesApplied": 1,
        "optimizerUpdatesSkipped": 0,
        "ampOverflowBatches": 0,
        "finiteGradientBatches": 1,
        "nonFiniteGradientBatches": 0,
        "preClipGradientNormMean": 1.0,
        "preClipGradientNormMax": 1.0,
        "preClipGradientNormP95": 1.0,
        "nan_ratio": 0.0,
        "masked_token_ratio": 0.0,
        "complete_token_ratio": 1.0,
        "partial_token_ratio": 0.0,
        "empty_token_ratio": 0.0,
        "input_pipeline_ms": 1.0,
        "missing_stats_ms": 1.0,
        "host_to_device_ms": 1.0,
        "train_step_ms": 1.0,
        "elapsed_ms": 4.0,
        "checkpoint_best": False,
        "should_stop": False,
        "best_selection_score": None,
    }
    record = build_training_record(
        metrics,
        recorded_at=1.0,
        job_id=job_id,
        attempt_id=attempt_id,
        attempt=attempt,
        model_ref=model_ref,
        data_contract_sha256=DATA_CONTRACT_SHA256,
        objective_config_sha256=objective_config_sha256(default_objective()),
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.10",
        git_commit="0" * 40,
        targets=default_objective().targets,
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
        data_contract_sha256=DATA_CONTRACT_SHA256,
        objective_config_sha256=objective_config_sha256(default_objective()),
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.10",
        git_commit="0" * 40,
        targets=default_objective().targets,
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
            "inputRows": 1,
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
