from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import BinaryIO, Protocol

from app.contracts.json_types import JsonObject
from app.contracts.metrics.v5 import (
    ARTIFACT_FORMAT,
    ARTIFACT_MEDIA_TYPE,
    build_training_record,
)
from app.contracts.semantic.v1 import ModelContract
from app.contracts.worker.v12 import validate_training_metrics_for_model
from app.service.application.telemetry.records import TrainingMetricIntervalRecord


class _MetricsSpool(Protocol):
    def staged_file(
        self,
        destination: str,
    ) -> AbstractContextManager[tuple[BinaryIO, str]]: ...


@dataclass(frozen=True, slots=True)
class StagedTrainingMetrics:
    path: str
    format: str
    media_type: str
    byte_count: int
    sha256: str
    row_count: int


def publish_training_metrics(
    spool: _MetricsSpool,
    destination: str,
    intervals: Sequence[TrainingMetricIntervalRecord],
    *,
    job_id: str,
    model_ref: str,
    semantic_digests: JsonObject,
    checkpoint_format: str,
    application_version: str,
    git_commit: str,
    targets: Sequence[str],
    model_contract: ModelContract,
) -> StagedTrainingMetrics:
    if not intervals:
        raise ValueError("fit run telemetry requires committed epoch metrics")
    expected_generations = list(range(1, len(intervals) + 1))
    if [item.generation for item in intervals] != expected_generations:
        raise ValueError("committed training metric generations are incomplete")

    digest = hashlib.sha256()
    byte_count = 0
    previous_step = -1
    identities: set[tuple[object, object, object]] = set()
    with spool.staged_file(destination) as (target, _):
        for interval in intervals:
            if interval.job_id != job_id:
                raise ValueError("committed training metrics job identity differs")
            metrics = interval.metrics
            validate_training_metrics_for_model(metrics, model_contract)
            if metrics.get("epoch") != interval.generation:
                raise ValueError(
                    "committed training metrics epoch differs from generation"
                )
            step = metrics.get("step")
            if isinstance(step, bool) or not isinstance(step, int):
                raise ValueError("committed training metrics step is invalid")
            if step < previous_step:
                raise ValueError("committed training metrics step is not monotonic")
            previous_step = step
            identity = (
                metrics.get("frame"),
                metrics.get("epoch"),
                step,
            )
            if identity in identities:
                raise ValueError("committed training metrics identity is duplicated")
            identities.add(identity)
            record = build_training_record(
                metrics,
                recorded_at=interval.recorded_at,
                job_id=job_id,
                attempt_id=interval.attempt_id,
                attempt=interval.attempt,
                model_ref=model_ref,
                semantic_digests=semantic_digests,
                checkpoint_format=checkpoint_format,
                application_version=application_version,
                git_commit=git_commit,
                targets=targets,
            )
            encoded = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            target.write(encoded)
            digest.update(encoded)
            byte_count += len(encoded)

    if byte_count <= 0 or os.path.getsize(destination) != byte_count:
        raise ValueError("published training metrics artifact is empty")
    return StagedTrainingMetrics(
        path=destination,
        format=ARTIFACT_FORMAT,
        media_type=ARTIFACT_MEDIA_TYPE,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
        row_count=len(intervals),
    )


__all__ = ["StagedTrainingMetrics", "publish_training_metrics"]
