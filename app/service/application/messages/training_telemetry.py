from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.service.domain.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class GetTrainingTelemetryReportQuery:
    owner_subject: str
    request_id: str
    model_ref: str
    page_size: int
    cursor: str | None


@dataclass(frozen=True, slots=True)
class GetGradientInteractionsQuery:
    owner_subject: str
    request_id: str
    model_ref: str
    epoch: int
    page_size: int
    cursor: str | None


@dataclass(frozen=True, slots=True)
class ProjectedTrainingTelemetry:
    state: Literal["pending", "unavailable", "available"]
    unavailable_reason: str | None = None
    report_identity: str | None = None
    run_document: JsonObject | None = None
    report_points: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingTelemetryReportResult:
    request_id: str
    model_ref: str
    state: Literal["pending", "unavailable", "available"]
    unavailable_reason: str | None = None
    producing_run_id: str | None = None
    semantic_digests: JsonObject | None = None
    coverage: JsonObject | None = None
    selection: JsonObject | None = None
    anchors: tuple[JsonObject, ...] = ()
    health_totals: JsonObject | None = None
    gradient_interactions: JsonObject | None = None
    epoch_items: tuple[JsonObject, ...] = ()
    next_cursor: str | None = None
    cursor_expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class GradientInteractionsResult:
    request_id: str
    model_ref: str
    epoch: int
    state: Literal["notCollected", "available"]
    reason: str | None = None
    producing_run_id: str | None = None
    components: tuple[JsonObject, ...] = ()
    pair_items: tuple[JsonObject, ...] = ()
    next_cursor: str | None = None
    cursor_expires_at: str | None = None


__all__ = [
    "GetGradientInteractionsQuery",
    "GetTrainingTelemetryReportQuery",
    "GradientInteractionsResult",
    "ProjectedTrainingTelemetry",
    "TrainingTelemetryReportResult",
]
