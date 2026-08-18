from __future__ import annotations

from dataclasses import dataclass

from app.service.domain.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class TrainingMetricIntervalRecord:
    job_id: str
    generation: int
    attempt: int
    attempt_id: str
    metrics: JsonObject
    recorded_at: float
    checkpoint_serialization_ms: float
    checkpoint_publication_ms: float


@dataclass(frozen=True, slots=True)
class FitRunSummarySource:
    job_id: str
    attempt_id: str
    attempt: int
    created_at: float
    first_input_committed_at: float
    input_closed_at: float
    worker_completed_at: float
    publication_boundary_at: float
    queue_wait_ms: float
    worker_startup_ms: float
    training_ms: float
    checkpoint_serialization_ms: float
    checkpoint_publication_ms: float
    attempt_count: int
    recovery_count: int
    input_payload_count: int
    input_rows: int
    input_bytes: int


@dataclass(frozen=True, slots=True)
class TrainingMetricsArtifactRecord:
    model_ref: str
    format: str
    media_type: str
    relative_path: str
    byte_count: int
    sha256: str
    row_count: int
    job_id: str
    attempt_id: str
    attempt: int
    application_version: str
    git_commit: str
    created_at: float


@dataclass(frozen=True, slots=True)
class FitRunSummaryArtifactRecord:
    model_ref: str
    format: str
    media_type: str
    relative_path: str
    byte_count: int
    sha256: str
    job_id: str
    attempt_id: str
    attempt: int
    application_version: str
    git_commit: str
    created_at: float


@dataclass(frozen=True, slots=True)
class MetricsOutboxRecord:
    training_metrics: TrainingMetricsArtifactRecord
    projection_version: str
    status: str
    cursor: int
    attempts: int
    next_attempt_at: float
    created_at: float
    run_summary: FitRunSummaryArtifactRecord | None = None


@dataclass(frozen=True, slots=True)
class TelemetryArtifactCleanup:
    job_id: str
    relative_paths: tuple[str, ...]


__all__ = [
    "FitRunSummaryArtifactRecord",
    "FitRunSummarySource",
    "MetricsOutboxRecord",
    "TelemetryArtifactCleanup",
    "TrainingMetricIntervalRecord",
    "TrainingMetricsArtifactRecord",
]
