from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from app.contracts.json_types import JsonObject
from app.contracts.metrics.fit_run.v6 import (
    SUMMARY_FORMAT,
    SUMMARY_MEDIA_TYPE,
    build_run_summary,
)
from app.service.application.telemetry.records import FitRunSummarySource

_COPY_CHUNK_BYTES = 1024 * 1024


class _SummarySpool(Protocol):
    def atomic_write_json(self, destination: str, document: JsonObject) -> str: ...


@dataclass(frozen=True, slots=True)
class StagedFitRunSummary:
    path: str
    format: str
    media_type: str
    byte_count: int
    sha256: str
    document: JsonObject


def publish_fit_run_summary(
    spool: _SummarySpool,
    destination: str,
    source: FitRunSummarySource,
    *,
    model_ref: str,
    semantic_digests: JsonObject,
    job_config_sha256: str,
    checkpoint_format: str,
    application_version: str,
    git_commit: str,
    targets: Sequence[str],
    initialization: JsonObject,
    terminal_checkpoint_serialization_ms: float,
    terminal_checkpoint_publication_ms: float,
) -> StagedFitRunSummary:
    serialization_ms = source.checkpoint_serialization_ms + _duration(
        terminal_checkpoint_serialization_ms,
        "terminal checkpoint serialization",
    )
    publication_ms = source.checkpoint_publication_ms + _duration(
        terminal_checkpoint_publication_ms,
        "terminal checkpoint publication",
    )
    document = build_run_summary(
        recorded_at=source.publication_boundary_at,
        job_id=source.job_id,
        attempt_id=source.attempt_id,
        attempt=source.attempt,
        model_ref=model_ref,
        semantic_digests=semantic_digests,
        job_config_sha256=job_config_sha256,
        checkpoint_format=checkpoint_format,
        application_version=application_version,
        git_commit=git_commit,
        targets=targets,
        initialization=initialization,
        milestones={
            "createdAt": _timestamp(source.created_at),
            "firstInputCommittedAt": _timestamp(source.first_input_committed_at),
            "inputClosedAt": _timestamp(source.input_closed_at),
            "workerCompletedAt": _timestamp(source.worker_completed_at),
            "publishedAt": _timestamp(source.publication_boundary_at),
        },
        durations={
            "firstInputWaitMs": _between(
                source.created_at,
                source.first_input_committed_at,
                "first input wait",
            ),
            "eofWaitMs": _between(
                source.first_input_committed_at,
                source.input_closed_at,
                "EOF wait",
            ),
            "queueWaitMs": _duration(source.queue_wait_ms, "queue wait"),
            "workerStartupMs": _duration(
                source.worker_startup_ms,
                "worker startup",
            ),
            "trainingMs": _duration(source.training_ms, "training"),
            "checkpointSerializationMs": serialization_ms,
            "checkpointPublicationMs": publication_ms,
            "modelPublicationMs": _between(
                source.worker_completed_at,
                source.publication_boundary_at,
                "model publication",
            ),
            "remoteFitMs": _between(
                source.created_at,
                source.publication_boundary_at,
                "remote fit",
            ),
        },
        counts={
            "attempts": source.attempt_count,
            "recoveries": source.recovery_count,
            "inputPayloads": source.input_payload_count,
            "inputChunks": source.input_chunks,
            "logicalRows": source.input_rows,
            "nativeRows": list(source.native_rows),
            "inputBytes": source.input_bytes,
        },
    )
    spool.atomic_write_json(destination, document)
    byte_count = os.path.getsize(destination)
    if byte_count <= 0:
        raise ValueError("fit run summary artifact is empty")
    return StagedFitRunSummary(
        path=destination,
        format=SUMMARY_FORMAT,
        media_type=SUMMARY_MEDIA_TYPE,
        byte_count=byte_count,
        sha256=_sha256_file(destination),
        document=document,
    )


def _between(start: float, end: float, label: str) -> float:
    return _duration((end - start) * 1000.0, label)


def _duration(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} duration must be finite and non-negative")
    return value


def _timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["StagedFitRunSummary", "publish_fit_run_summary"]
