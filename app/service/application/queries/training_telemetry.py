from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import NoReturn, cast

from app.service.application.messages.training_telemetry import (
    GetGradientInteractionsQuery,
    GetTrainingTelemetryReportQuery,
    GradientInteractionsResult,
    ProjectedTrainingTelemetry,
    TrainingTelemetryReportResult,
)
from app.service.application.ports.model_catalog import (
    CatalogMetadataVerifier,
    CatalogModelNotFound,
    ModelCatalogStore,
)
from app.service.application.ports.training_telemetry import (
    TrainingTelemetryIntegrityError,
    TrainingTelemetrySource,
    TrainingTelemetryStoredMetadataError,
)
from app.service.application.services.training_telemetry_cursor import (
    InvalidTrainingTelemetryCursor,
    TrainingTelemetryCursor,
    TrainingTelemetryCursorCodec,
)
from app.service.application.services.training_telemetry_snapshot import (
    TrainingTelemetrySnapshotStore,
)
from app.service.domain.json_types import JsonObject, JsonValue
from app.service.domain.records import PublishedModelRecord

_HEALTH_METRICS = {
    "training.batches.completed": "trainingBatchesCompleted",
    "training.optimizer.updates.applied": "optimizerUpdatesApplied",
    "training.optimizer.updates.skipped": "optimizerUpdatesSkipped",
    "training.amp.overflow_batches": "ampOverflowBatches",
    "training.gradient.batches.finite": "finiteGradientBatches",
    "training.gradient.batches.non_finite": "nonFiniteGradientBatches",
}
_GRADIENT_COMPONENT_METRIC = "training.gradient.component.norm"
_GRADIENT_COSINE_METRIC = "training.gradient.pair.cosine"
_GRADIENT_NEGATIVE_METRIC = "training.gradient.pair.negative_fraction"


@dataclass(frozen=True, slots=True)
class _Component:
    identity: str
    operator: str
    target_identity: str | None
    target_index: int | None


@dataclass(frozen=True, slots=True)
class _ValidatedReport:
    model_ref: str
    producing_run_id: str
    report_identity: str
    semantic_digests: JsonObject
    epochs: tuple[JsonObject, ...]
    selection: JsonObject
    anchors: tuple[JsonObject, ...]
    health_totals: JsonObject
    gradient_summary: JsonObject
    collected_epochs: frozenset[int]
    components: tuple[_Component, ...]
    diagnostics_configured: bool


@dataclass(frozen=True, slots=True)
class _RetainedReport:
    model_ref: str
    producing_run_id: str
    report_identity: str
    semantic_digests: JsonObject
    epochs: tuple[JsonObject, ...]
    selection: JsonObject
    anchors: tuple[JsonObject, ...]
    health_totals: JsonObject
    gradient_summary: JsonObject
    components: tuple[_Component, ...]


@dataclass(frozen=True, slots=True)
class _RetainedGradientInteractions:
    model_ref: str
    producing_run_id: str
    epoch: int
    identity: str
    components: tuple[JsonObject, ...]
    pairs: tuple[JsonObject, ...]


class GetTrainingTelemetryReport:
    def __init__(
        self,
        store: ModelCatalogStore,
        source: TrainingTelemetrySource,
        snapshots: TrainingTelemetrySnapshotStore,
        *,
        metadata_verifier: CatalogMetadataVerifier,
        cursor_ttl_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._source = source
        self._snapshots = snapshots
        self._metadata_verifier = metadata_verifier
        self._cursor_ttl_seconds = cursor_ttl_seconds
        self._clock = clock
        self._codec: TrainingTelemetryCursorCodec | None = None
        self._codec_lock = threading.Lock()

    def execute(
        self,
        query: GetTrainingTelemetryReportQuery,
    ) -> TrainingTelemetryReportResult:
        now = self._clock().astimezone(UTC)
        model = _visible_model(
            self._store,
            self._metadata_verifier,
            query.owner_subject,
            query.model_ref,
        )
        producing_run_id = _producing_run_id(model)
        cursor = None
        if query.cursor is not None:
            cursor = self._cursor_codec().decode(
                query.owner_subject,
                query.cursor,
                operation="report",
                model_ref=query.model_ref,
                producing_run_id=producing_run_id,
                page_size=query.page_size,
                epoch=None,
                now=now,
            )
        cached = (
            None
            if cursor is None
            else self._snapshots.get(
                _report_snapshot_key(
                    query.owner_subject,
                    model.model_ref,
                    producing_run_id,
                    cursor.report_identity,
                ),
                now=now,
            )
        )
        if cached is not None and not isinstance(cached, _RetainedReport):
            raise RuntimeError("retained report snapshot has invalid type")
        report = cached
        if cursor is not None and report is None:
            raise InvalidTrainingTelemetryCursor
        if report is None:
            projected = self._source.load_report(
                model_ref=model.model_ref,
                producing_run_id=producing_run_id,
            )
            if projected.state == "pending":
                return TrainingTelemetryReportResult(
                    query.request_id,
                    model.model_ref,
                    "pending",
                )
            if projected.state == "unavailable":
                return TrainingTelemetryReportResult(
                    query.request_id,
                    model.model_ref,
                    "unavailable",
                    unavailable_reason=projected.unavailable_reason,
                )
            validated = _validate_report(model, projected)
            report = _retained_report(validated)
        after_epoch = 0 if cursor is None else cast(int, cursor.after_epoch)
        if after_epoch >= len(report.epochs):
            raise InvalidTrainingTelemetryCursor
        page = report.epochs[after_epoch : after_epoch + query.page_size]
        terminal = after_epoch + len(page) == len(report.epochs)
        expires_at = (
            now + timedelta(seconds=self._cursor_ttl_seconds)
            if cursor is None
            else cursor.expires_at
        )
        next_cursor = None
        cursor_expires_at = None
        if not terminal:
            retained = self._snapshots.admit(
                _report_snapshot_key(
                    query.owner_subject,
                    model.model_ref,
                    report.producing_run_id,
                    report.report_identity,
                ),
                report,
                _report_snapshot_projection(report),
                operation="report",
                expires_at=expires_at,
                now=now,
            )
            if not isinstance(retained, _RetainedReport):
                raise RuntimeError("retained report snapshot has invalid type")
            report = retained
            last_epoch = _integer(page[-1], "epoch")
            next_cursor = self._cursor_codec().encode(
                query.owner_subject,
                TrainingTelemetryCursor(
                    operation="report",
                    model_ref=model.model_ref,
                    producing_run_id=report.producing_run_id,
                    report_identity=report.report_identity,
                    page_size=query.page_size,
                    after_epoch=last_epoch,
                    epoch=None,
                    after_pair=None,
                    expires_at=expires_at,
                ),
            )
            cursor_expires_at = _timestamp(expires_at)
        return TrainingTelemetryReportResult(
            request_id=query.request_id,
            model_ref=model.model_ref,
            state="available",
            producing_run_id=report.producing_run_id,
            semantic_digests=dict(report.semantic_digests),
            layout=_report_layout(report.components),
            coverage={
                "completedEpochs": len(report.epochs),
            },
            selection=dict(report.selection),
            anchors=tuple(
                _compact_anchor(item, report.components)
                for item in report.anchors
            ),
            health_totals=dict(report.health_totals),
            gradient_interactions=dict(report.gradient_summary),
            epoch_items=tuple(
                _compact_epoch(item, report.components)
                for item in page
            ),
            next_cursor=next_cursor,
            cursor_expires_at=cursor_expires_at,
        )

    def _cursor_codec(self) -> TrainingTelemetryCursorCodec:
        with self._codec_lock:
            if self._codec is None:
                self._codec = TrainingTelemetryCursorCodec(
                    self._store.cursor_signing_key(),
                    boot_identity=self._snapshots.boot_identity,
                )
            return self._codec


class GetGradientInteractions:
    def __init__(
        self,
        store: ModelCatalogStore,
        source: TrainingTelemetrySource,
        snapshots: TrainingTelemetrySnapshotStore,
        *,
        metadata_verifier: CatalogMetadataVerifier,
        cursor_ttl_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._source = source
        self._snapshots = snapshots
        self._metadata_verifier = metadata_verifier
        self._cursor_ttl_seconds = cursor_ttl_seconds
        self._clock = clock
        self._codec: TrainingTelemetryCursorCodec | None = None
        self._codec_lock = threading.Lock()

    def execute(
        self,
        query: GetGradientInteractionsQuery,
    ) -> GradientInteractionsResult:
        now = self._clock().astimezone(UTC)
        model = _visible_model(
            self._store,
            self._metadata_verifier,
            query.owner_subject,
            query.model_ref,
        )
        producing_run_id = _producing_run_id(model)
        cursor = None
        if query.cursor is not None:
            cursor = self._cursor_codec().decode(
                query.owner_subject,
                query.cursor,
                operation="gradientInteractions",
                model_ref=query.model_ref,
                producing_run_id=producing_run_id,
                page_size=query.page_size,
                epoch=query.epoch,
                now=now,
            )
        cached = (
            None
            if cursor is None
            else self._snapshots.get(
                _gradient_snapshot_key(
                    query.owner_subject,
                    model.model_ref,
                    producing_run_id,
                    query.epoch,
                    cursor.report_identity,
                ),
                now=now,
            )
        )
        if cached is not None and not isinstance(
            cached,
            _RetainedGradientInteractions,
        ):
            raise RuntimeError("retained gradient snapshot has invalid type")
        snapshot = cached
        if cursor is not None and snapshot is None:
            raise InvalidTrainingTelemetryCursor
        if snapshot is None:
            projected = self._source.load_report(
                model_ref=model.model_ref,
                producing_run_id=producing_run_id,
            )
            if projected.state != "available":
                return _not_collected(
                    query,
                    model,
                    _diagnostics_configured(model),
                )
            report = _validate_report(model, projected)
            if not report.diagnostics_configured:
                return _not_collected(query, model, False)
            if query.epoch not in report.collected_epochs:
                return _not_collected(query, model, True)
            points = self._source.load_gradient_points(
                model_ref=model.model_ref,
                producing_run_id=report.producing_run_id,
                epoch=query.epoch,
            )
            components, pairs, gradient_identity = _validate_gradient_points(
                model,
                report,
                query.epoch,
                points,
            )
            snapshot = _RetainedGradientInteractions(
                model_ref=model.model_ref,
                producing_run_id=report.producing_run_id,
                epoch=query.epoch,
                identity=gradient_identity,
                components=components,
                pairs=pairs,
            )
        start = 0
        if cursor is not None:
            after_pair = cast(tuple[str, str], cursor.after_pair)
            identities = tuple(
                (
                    cast(str, item["leftComponentIdentity"]),
                    cast(str, item["rightComponentIdentity"]),
                )
                for item in snapshot.pairs
            )
            try:
                start = identities.index(after_pair) + 1
            except ValueError as exc:
                raise InvalidTrainingTelemetryCursor from exc
        page = snapshot.pairs[start : start + query.page_size]
        terminal = start + len(page) == len(snapshot.pairs)
        expires_at = (
            now + timedelta(seconds=self._cursor_ttl_seconds)
            if cursor is None
            else cursor.expires_at
        )
        next_cursor = None
        cursor_expires_at = None
        if not terminal:
            retained = self._snapshots.admit(
                _gradient_snapshot_key(
                    query.owner_subject,
                    model.model_ref,
                    snapshot.producing_run_id,
                    query.epoch,
                    snapshot.identity,
                ),
                snapshot,
                _gradient_snapshot_projection(snapshot),
                operation="gradientInteractions",
                expires_at=expires_at,
                now=now,
            )
            if not isinstance(retained, _RetainedGradientInteractions):
                raise RuntimeError("retained gradient snapshot has invalid type")
            snapshot = retained
            last = page[-1]
            pair_identity = (
                cast(str, last["leftComponentIdentity"]),
                cast(str, last["rightComponentIdentity"]),
            )
            next_cursor = self._cursor_codec().encode(
                query.owner_subject,
                TrainingTelemetryCursor(
                    operation="gradientInteractions",
                    model_ref=model.model_ref,
                    producing_run_id=snapshot.producing_run_id,
                    report_identity=snapshot.identity,
                    page_size=query.page_size,
                    after_epoch=None,
                    epoch=query.epoch,
                    after_pair=pair_identity,
                    expires_at=expires_at,
                ),
            )
            cursor_expires_at = _timestamp(expires_at)
        return GradientInteractionsResult(
            request_id=query.request_id,
            model_ref=model.model_ref,
            epoch=query.epoch,
            state="available",
            producing_run_id=snapshot.producing_run_id,
            components=snapshot.components,
            pair_items=tuple(dict(item) for item in page),
            next_cursor=next_cursor,
            cursor_expires_at=cursor_expires_at,
        )

    def _cursor_codec(self) -> TrainingTelemetryCursorCodec:
        with self._codec_lock:
            if self._codec is None:
                self._codec = TrainingTelemetryCursorCodec(
                    self._store.cursor_signing_key(),
                    boot_identity=self._snapshots.boot_identity,
                )
            return self._codec


def _report_snapshot_key(
    owner_subject: str,
    model_ref: str,
    producing_run_id: str,
    report_identity: str,
) -> tuple[str, ...]:
    return (
        "report",
        owner_subject,
        model_ref,
        producing_run_id,
        report_identity,
    )


def _gradient_snapshot_key(
    owner_subject: str,
    model_ref: str,
    producing_run_id: str,
    epoch: int,
    report_identity: str,
) -> tuple[str, ...]:
    return (
        "gradientInteractions",
        owner_subject,
        model_ref,
        producing_run_id,
        str(epoch),
        report_identity,
    )


def _retained_report(report: _ValidatedReport) -> _RetainedReport:
    return _RetainedReport(
        model_ref=report.model_ref,
        producing_run_id=report.producing_run_id,
        report_identity=report.report_identity,
        semantic_digests=report.semantic_digests,
        epochs=report.epochs,
        selection=report.selection,
        anchors=report.anchors,
        health_totals=report.health_totals,
        gradient_summary=report.gradient_summary,
        components=report.components,
    )


def _report_snapshot_projection(report: _RetainedReport) -> JsonValue:
    return {
        "snapshotType": "report",
        "modelRef": report.model_ref,
        "producingRunId": report.producing_run_id,
        "reportIdentity": report.report_identity,
        "semanticDigests": report.semantic_digests,
        "epochs": list(report.epochs),
        "selection": report.selection,
        "anchors": list(report.anchors),
        "healthTotals": report.health_totals,
        "gradientInteractions": report.gradient_summary,
        "layout": _report_layout(report.components),
    }


def _gradient_snapshot_projection(
    snapshot: _RetainedGradientInteractions,
) -> JsonValue:
    return {
        "snapshotType": "gradientInteractions",
        "modelRef": snapshot.model_ref,
        "producingRunId": snapshot.producing_run_id,
        "epoch": snapshot.epoch,
        "reportIdentity": snapshot.identity,
        "components": list(snapshot.components),
        "pairs": list(snapshot.pairs),
    }


def _report_layout(components: Sequence[_Component]) -> JsonObject:
    direct = [component for component in components if component.target_index is not None]
    auxiliary = [component for component in components if component.target_index is None]
    return {
        "targetIdentities": [
            cast(str, component.target_identity)
            for component in direct
        ],
        "directComponents": [
            {
                "identity": component.identity,
                "operator": component.operator,
            }
            for component in direct
        ],
        "auxiliaryComponents": [
            {
                "identity": component.identity,
                "operator": component.operator,
            }
            for component in auxiliary
        ],
    }


def _compact_epoch(
    epoch: Mapping[str, object],
    components: Sequence[_Component],
) -> JsonObject:
    direct_count = sum(component.target_index is not None for component in components)
    direct_losses = _sequence(epoch, "directLosses")
    auxiliary_losses = _sequence(epoch, "auxiliaryLosses")
    target_metrics = _sequence(epoch, "targetMetrics")
    if (
        len(direct_losses) != direct_count
        or len(auxiliary_losses) != len(components) - direct_count
        or len(target_metrics) != direct_count
    ):
        _integrity("/epochs", "epoch layout differs from report layout")
    return {
        "epoch": _integer(epoch, "epoch"),
        "globalStep": _integer(epoch, "globalStep"),
        "attempt": _integer(epoch, "attempt"),
        "totalLoss": _number(epoch, "totalLoss"),
        "selectionScore": _optional_number(epoch, "selectionScore"),
        "directLosses": [
            _number(cast(Mapping[str, object], item), "value")
            for item in direct_losses
        ],
        "auxiliaryLosses": [
            _number(cast(Mapping[str, object], item), "value")
            for item in auxiliary_losses
        ],
        "targetMetrics": [
            {
                "mae": _number(cast(Mapping[str, object], item), "mae"),
                "rmse": _number(cast(Mapping[str, object], item), "rmse"),
            }
            for item in target_metrics
        ],
        "health": dict(cast(JsonObject, epoch["health"])),
        "gradientInteractionsCollected": bool(
            epoch["gradientInteractionsCollected"]
        ),
    }


def _compact_anchor(
    anchor: Mapping[str, object],
    components: Sequence[_Component],
) -> JsonObject:
    compact = _compact_epoch(anchor, components)
    roles = _sequence(anchor, "roles")
    if not all(isinstance(role, str) for role in roles):
        _integrity("/roles", "anchor roles are invalid")
    compact["roles"] = cast(list[JsonValue], list(roles))
    return compact


def _visible_model(
    store: ModelCatalogStore,
    verifier: CatalogMetadataVerifier,
    owner_subject: str,
    model_ref: str,
) -> PublishedModelRecord:
    entry = store.get_model(owner_subject, model_ref)
    if entry is None:
        raise CatalogModelNotFound(model_ref)
    verifier.verify(entry.model)
    return entry.model


def _validate_report(
    model: PublishedModelRecord,
    projected: ProjectedTrainingTelemetry,
) -> _ValidatedReport:
    run = projected.run_document
    identity = projected.report_identity
    if (
        run is None
        or identity is None
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        _integrity("", "complete telemetry projection is incomplete")
    run_id = _producing_run_id(model)
    if run.get("runId") != run_id:
        _integrity("/producingRunId", "telemetry producing run differs")
    if run.get("modelRef") != model.model_ref:
        _integrity("/modelRef", "telemetry model reference differs")
    if run.get("semanticDigests") != model.semantic_digests:
        _integrity("/semanticDigests", "telemetry semantic digests differ")
    metadata = model.metadata
    if run.get("jobConfigSha256") != metadata.get("jobConfigSha256"):
        _integrity("/jobConfigSha256", "telemetry job configuration differs")
    if run.get("checkpointFormat") != metadata.get("format"):
        _integrity("/checkpointFormat", "telemetry checkpoint format differs")

    targets, direct, auxiliary = _model_layout(model)
    run_targets = run.get("targets")
    expected_targets = [
        {"index": index, "identity": target}
        for index, target in enumerate(targets)
    ]
    if run_targets != expected_targets:
        _integrity("/targets", "telemetry target layout differs")
    completed_epochs = _metadata_positive_integer(model, "progress", "completedEpochs")
    expected_final_step = _metadata_nonnegative_integer(model, "progress", "globalStep")
    selection = _selection(model, completed_epochs)
    diagnostics_configured = _diagnostics_configured(model)
    components = (*direct, *auxiliary)

    grouped: dict[int, list[JsonObject]] = {}
    seen: set[tuple[object, ...]] = set()
    for point_index, point in enumerate(projected.report_points):
        if point.get("runId") != run_id or point.get("modelRef") != model.model_ref:
            _integrity(f"/points/{point_index}", "telemetry point identity differs")
        if point.get("semanticDigests") != model.semantic_digests:
            _integrity(
                f"/points/{point_index}/semanticDigests",
                "telemetry point semantic digests differ",
            )
        epoch = _integer(point, "epoch")
        if not 1 <= epoch <= completed_epochs:
            _integrity(f"/points/{point_index}/epoch", "telemetry epoch is out of range")
        key = _point_key(point)
        if key in seen:
            _integrity(f"/points/{point_index}", "duplicate telemetry observation")
        seen.add(key)
        grouped.setdefault(epoch, []).append(point)

    epochs: list[JsonObject] = []
    collected_epochs: set[int] = set()
    previous_step = -1
    health_totals = {field: 0 for field in _HEALTH_METRICS.values()}
    for epoch in range(1, completed_epochs + 1):
        points = grouped.get(epoch)
        if points is None:
            _integrity("/coverage", f"telemetry epoch {epoch} is missing")
        item, collected = _epoch_document(
            epoch,
            points,
            targets,
            direct,
            auxiliary,
            selection_enabled=cast(bool, selection["enabled"]),
        )
        step = _integer(item, "globalStep")
        if step <= previous_step:
            _integrity(f"/epochs/{epoch - 1}/globalStep", "global step is not increasing")
        previous_step = step
        health = cast(JsonObject, item["health"])
        for field in health_totals:
            health_totals[field] += _integer(health, field)
        if collected:
            collected_epochs.add(epoch)
        epochs.append(item)
    if previous_step != expected_final_step:
        _integrity("/coverage", "telemetry final global step differs")
    if not diagnostics_configured and collected_epochs:
        _integrity("/gradientInteractions", "unconfigured gradient telemetry is present")

    published_epoch = cast(int, selection["publishedEpoch"])
    gradient_summary = _gradient_summary(
        diagnostics_configured,
        frozenset(collected_epochs),
        published_epoch,
    )
    anchors = _anchors(tuple(epochs), selection)
    return _ValidatedReport(
        model_ref=model.model_ref,
        producing_run_id=run_id,
        report_identity=identity,
        semantic_digests=dict(model.semantic_digests),
        epochs=tuple(epochs),
        selection=selection,
        anchors=anchors,
        health_totals=cast(JsonObject, health_totals),
        gradient_summary=gradient_summary,
        collected_epochs=frozenset(collected_epochs),
        components=tuple(components),
        diagnostics_configured=diagnostics_configured,
    )


def _epoch_document(
    epoch: int,
    points: Sequence[JsonObject],
    targets: tuple[str, ...],
    direct: tuple[_Component, ...],
    auxiliary: tuple[_Component, ...],
    *,
    selection_enabled: bool,
) -> tuple[JsonObject, bool]:
    attempts = {_integer(point, "attempt") for point in points}
    steps = {_integer(point, "step") for point in points}
    if len(attempts) != 1 or len(steps) != 1:
        _integrity(f"/epochs/{epoch - 1}", "epoch point identities differ")
    scalar: dict[str, float] = {}
    direct_values: dict[str, JsonObject] = {}
    auxiliary_values: dict[str, JsonObject] = {}
    target_values: dict[tuple[str, str], float] = {}
    gradient_components: set[str] = set()
    component_by_id = {item.identity: item for item in (*direct, *auxiliary)}

    for point in points:
        metric = _mapping(point, "metric")
        name = _string(metric, "name")
        value = _number(metric, "value")
        target = _optional_mapping(point, "target")
        component = _optional_mapping(point, "component")
        pair = _optional_mapping(point, "pair")
        if pair is not None:
            _integrity(f"/epochs/{epoch - 1}", "gradient pair leaked into report projection")
        if name in {"training.loss.total", "training.selection.score", *_HEALTH_METRICS}:
            if target is not None or component is not None:
                _integrity(f"/epochs/{epoch - 1}", "scalar metric has context")
            scalar[name] = value
            continue
        if name == "training.loss.direct":
            candidate = _component_context(component_by_id, target, component, direct)
            direct_values[candidate.identity] = {
                "componentIdentity": candidate.identity,
                "operator": candidate.operator,
                "targetIdentity": candidate.target_identity,
                "targetIndex": candidate.target_index,
                "value": value,
            }
            continue
        if name == "training.loss.auxiliary":
            candidate = _component_context(component_by_id, target, component, auxiliary)
            auxiliary_values[candidate.identity] = {
                "componentIdentity": candidate.identity,
                "operator": candidate.operator,
                "value": value,
            }
            continue
        if name in {"training.target.mae", "training.target.rmse"}:
            if target is None or component is not None:
                _integrity(f"/epochs/{epoch - 1}", "target metric context differs")
            target_identity = _string(target, "identity")
            target_index = _integer(target, "index")
            if target_index >= len(targets) or targets[target_index] != target_identity:
                _integrity(f"/epochs/{epoch - 1}/targetMetrics", "target metric does not resolve")
            if value < 0:
                _integrity(
                    f"/epochs/{epoch - 1}/targetMetrics",
                    "target error observation is negative",
                )
            target_values[(name, target_identity)] = value
            continue
        if name == _GRADIENT_COMPONENT_METRIC:
            candidate = _component_context(
                component_by_id,
                target,
                component,
                (*direct, *auxiliary),
            )
            gradient_components.add(candidate.identity)
            continue
        _integrity(f"/epochs/{epoch - 1}", "unsupported report metric observation")

    required_scalars = {"training.loss.total", *_HEALTH_METRICS}
    if selection_enabled:
        required_scalars.add("training.selection.score")
    if set(scalar) != required_scalars:
        _integrity(f"/epochs/{epoch - 1}", "epoch scalar observations are incomplete")
    if set(direct_values) != {item.identity for item in direct}:
        _integrity(f"/epochs/{epoch - 1}/directLosses", "direct observations are incomplete")
    if set(auxiliary_values) != {item.identity for item in auxiliary}:
        _integrity(f"/epochs/{epoch - 1}/auxiliaryLosses", "auxiliary observations are incomplete")
    expected_target_values = {
        (metric, target)
        for metric in ("training.target.mae", "training.target.rmse")
        for target in targets
    }
    if set(target_values) != expected_target_values:
        _integrity(f"/epochs/{epoch - 1}/targetMetrics", "target observations are incomplete")
    expected_components = {item.identity for item in (*direct, *auxiliary)}
    if gradient_components and gradient_components != expected_components:
        _integrity(
            f"/epochs/{epoch - 1}/gradientInteractions",
            "gradient component observations are incomplete",
        )
    health: JsonObject = {
        field: _exact_counter(scalar[name], epoch, field)
        for name, field in _HEALTH_METRICS.items()
    }
    _validate_health(health, f"/epochs/{epoch - 1}/health")
    target_metrics = [
        {
            "targetIdentity": target,
            "targetIndex": index,
            "mae": target_values[("training.target.mae", target)],
            "rmse": target_values[("training.target.rmse", target)],
        }
        for index, target in enumerate(targets)
    ]
    document: JsonObject = {
        "epoch": epoch,
        "globalStep": steps.pop(),
        "attempt": attempts.pop(),
        "totalLoss": scalar["training.loss.total"],
        "selectionScore": scalar.get("training.selection.score"),
        "directLosses": [direct_values[item.identity] for item in direct],
        "auxiliaryLosses": [
            auxiliary_values[item.identity] for item in auxiliary
        ],
        "targetMetrics": cast(list[JsonValue], target_metrics),
        "health": health,
        "gradientInteractionsCollected": bool(gradient_components),
    }
    return document, bool(gradient_components)


def _validate_gradient_points(
    model: PublishedModelRecord,
    report: _ValidatedReport,
    epoch: int,
    points: Sequence[JsonObject],
) -> tuple[tuple[JsonObject, ...], tuple[JsonObject, ...], str]:
    component_by_id = {item.identity: item for item in report.components}
    component_values: dict[str, JsonObject] = {}
    pair_values: dict[tuple[str, str], dict[str, float]] = {}
    digests: list[str] = []
    seen: set[tuple[object, ...]] = set()
    order = {item.identity: index for index, item in enumerate(report.components)}
    epoch_document = report.epochs[epoch - 1]
    expected_attempt = _integer(epoch_document, "attempt")
    expected_step = _integer(epoch_document, "globalStep")
    for point_index, point in enumerate(points):
        if (
            point.get("runId") != report.producing_run_id
            or point.get("modelRef") != model.model_ref
            or point.get("semanticDigests") != model.semantic_digests
            or point.get("epoch") != epoch
            or point.get("attempt") != expected_attempt
            or point.get("step") != expected_step
        ):
            _integrity(f"/gradientPoints/{point_index}", "gradient point identity differs")
        key = _point_key(point)
        if key in seen:
            _integrity(f"/gradientPoints/{point_index}", "duplicate gradient observation")
        seen.add(key)
        document_digest = point.get("documentSha256")
        if not isinstance(document_digest, str):
            _integrity(f"/gradientPoints/{point_index}", "gradient point digest is missing")
        digests.append(document_digest)
        metric = _mapping(point, "metric")
        name = _string(metric, "name")
        value = _number(metric, "value")
        if name == _GRADIENT_COMPONENT_METRIC:
            candidate = _component_context(
                component_by_id,
                _optional_mapping(point, "target"),
                _optional_mapping(point, "component"),
                report.components,
            )
            component_values[candidate.identity] = {
                "componentIdentity": candidate.identity,
                **(
                    {}
                    if candidate.target_identity is None
                    else {
                        "targetIdentity": candidate.target_identity,
                        "targetIndex": candidate.target_index,
                    }
                ),
                "meanNorm": value,
            }
            if value < 0:
                _integrity(
                    f"/gradientPoints/{point_index}/metric/value",
                    "gradient component norm is negative",
                )
            continue
        if name not in {_GRADIENT_COSINE_METRIC, _GRADIENT_NEGATIVE_METRIC}:
            _integrity(f"/gradientPoints/{point_index}", "unsupported gradient metric")
        if _optional_mapping(point, "target") is not None or _optional_mapping(point, "component") is not None:
            _integrity(f"/gradientPoints/{point_index}", "gradient pair has component context")
        pair = _optional_mapping(point, "pair")
        if pair is None:
            _integrity(f"/gradientPoints/{point_index}", "gradient pair context is missing")
        left = _string(pair, "leftComponentIdentity")
        right = _string(pair, "rightComponentIdentity")
        if left not in order or right not in order or order[left] >= order[right]:
            _integrity(f"/gradientPoints/{point_index}/pair", "gradient pair orientation differs")
        if name == _GRADIENT_COSINE_METRIC and not -1 <= value <= 1:
            _integrity(
                f"/gradientPoints/{point_index}/metric/value",
                "gradient cosine is outside [-1,1]",
            )
        if name == _GRADIENT_NEGATIVE_METRIC and not 0 <= value <= 1:
            _integrity(
                f"/gradientPoints/{point_index}/metric/value",
                "negative cosine fraction is outside [0,1]",
            )
        pair_values.setdefault((left, right), {})[name] = value

    if set(component_values) != set(component_by_id):
        _integrity("/components", "gradient component observations are incomplete")
    pairs: list[JsonObject] = []
    for (left, right), values in pair_values.items():
        if set(values) != {_GRADIENT_COSINE_METRIC, _GRADIENT_NEGATIVE_METRIC}:
            _integrity("/pairs", "gradient pair observations are incomplete")
        pairs.append({
            "leftComponentIdentity": left,
            "rightComponentIdentity": right,
            "meanCosine": values[_GRADIENT_COSINE_METRIC],
            "negativeCosineFraction": values[_GRADIENT_NEGATIVE_METRIC],
        })
    pairs.sort(key=lambda item: (
        cast(str, item["leftComponentIdentity"]),
        cast(str, item["rightComponentIdentity"]),
    ))
    identity = hashlib.sha256(
        (report.report_identity + "\0" + "\0".join(sorted(digests))).encode("ascii")
    ).hexdigest()
    return (
        tuple(component_values[item.identity] for item in report.components),
        tuple(pairs),
        identity,
    )


def _model_layout(
    model: PublishedModelRecord,
) -> tuple[tuple[str, ...], tuple[_Component, ...], tuple[_Component, ...]]:
    contract = cast(Mapping[str, object], model.model_contract)
    target_contract = _mapping(contract, "targetContract")
    slot_values = _sequence(target_contract, "slots")
    targets = tuple(_string(cast(Mapping[str, object], slot), "identity") for slot in slot_values)
    objective = _mapping(contract, "objective")
    direct_values = _sequence(objective, "directComponents")
    raw_auxiliary = objective.get("auxiliaryComponents", ())
    if not isinstance(raw_auxiliary, Sequence) or isinstance(
        raw_auxiliary,
        (str, bytes),
    ):
        _integrity("/objective/auxiliaryComponents", "objective array is invalid")
    auxiliary_values = cast(Sequence[object], raw_auxiliary)
    direct = tuple(
        _Component(
            identity=_string(cast(Mapping[str, object], item), "identity"),
            operator=_string(cast(Mapping[str, object], item), "operator"),
            target_identity=targets[index],
            target_index=index,
        )
        for index, item in enumerate(direct_values)
    )
    auxiliary = tuple(
        _Component(
            identity=_string(cast(Mapping[str, object], item), "identity"),
            operator=_string(cast(Mapping[str, object], item), "operator"),
            target_identity=None,
            target_index=None,
        )
        for item in auxiliary_values
    )
    return targets, direct, auxiliary


def _component_context(
    component_by_id: Mapping[str, _Component],
    target: Mapping[str, object] | None,
    component: Mapping[str, object] | None,
    allowed: Sequence[_Component],
) -> _Component:
    if component is None:
        _integrity("/component", "component context is missing")
    identity = _string(component, "identity")
    candidate = component_by_id.get(identity)
    if candidate is None or candidate not in allowed or component.get("operator") != candidate.operator:
        _integrity("/component", "component context does not resolve")
    if candidate.target_identity is None:
        if target is not None:
            _integrity("/target", "auxiliary component has target context")
    elif (
        target is None
        or target.get("identity") != candidate.target_identity
        or target.get("index") != candidate.target_index
    ):
        _integrity("/target", "direct component target context differs")
    return candidate


def _selection(model: PublishedModelRecord, completed_epochs: int) -> JsonObject:
    selection = _metadata_mapping(model, "selection")
    enabled = selection.get("enabled")
    best_epoch = selection.get("bestEpoch")
    source = selection.get("source")
    if enabled is False and best_epoch is None and source == "last_epoch":
        return {
            "enabled": False,
            "bestEpoch": None,
            "publishedEpoch": completed_epochs,
            "publishedSource": "last_epoch",
        }
    if (
        enabled is True
        and isinstance(best_epoch, int)
        and not isinstance(best_epoch, bool)
        and 1 <= best_epoch <= completed_epochs
        and source == "best_direct_selection_score"
    ):
        return {
            "enabled": True,
            "bestEpoch": best_epoch,
            "publishedEpoch": best_epoch,
            "publishedSource": "best_direct_selection_score",
        }
    raise TrainingTelemetryStoredMetadataError(
        "/selection",
        "stored selection metadata is inconsistent",
    )


def _anchors(
    epochs: tuple[JsonObject, ...],
    selection: JsonObject,
) -> tuple[JsonObject, ...]:
    roles: dict[int, list[str]] = {1: ["first"]}
    best = selection["bestEpoch"]
    published = cast(int, selection["publishedEpoch"])
    if isinstance(best, int):
        roles.setdefault(best, []).append("best")
    roles.setdefault(published, []).append("published")
    result: list[JsonObject] = []
    for epoch, epoch_roles in sorted(roles.items()):
        anchor = dict(epochs[epoch - 1])
        anchor["roles"] = cast(list[JsonValue], epoch_roles)
        result.append(anchor)
    return tuple(result)


def _gradient_summary(
    configured: bool,
    collected: frozenset[int],
    published_epoch: int,
) -> JsonObject:
    if not configured:
        return {"state": "notConfigured"}
    if not collected:
        return {"state": "configuredWithoutObservations"}
    if published_epoch in collected:
        default = published_epoch
    else:
        previous = [epoch for epoch in collected if epoch < published_epoch]
        following = [epoch for epoch in collected if epoch > published_epoch]
        default = max(previous) if previous else min(following)
    return {
        "state": "available",
        "collectedEpochCount": len(collected),
        "defaultEpoch": default,
        "publishedEpochCollected": published_epoch in collected,
    }


def _not_collected(
    query: GetGradientInteractionsQuery,
    model: PublishedModelRecord,
    configured: bool,
) -> GradientInteractionsResult:
    return GradientInteractionsResult(
        request_id=query.request_id,
        model_ref=model.model_ref,
        epoch=query.epoch,
        state="notCollected",
        reason=(
            "NO_OBSERVATIONS_FOR_EPOCH"
            if configured
            else "DIAGNOSTICS_NOT_CONFIGURED"
        ),
    )


def _diagnostics_configured(model: PublishedModelRecord) -> bool:
    diagnostics = _metadata_mapping(model, "diagnostics")
    return diagnostics.get("gradientInteractions") is not None


def _producing_run_id(model: PublishedModelRecord) -> str:
    value = model.metadata.get("jobId")
    if not isinstance(value, str):
        raise TrainingTelemetryStoredMetadataError(
            "/producingRunId",
            "stored producing run is unavailable",
        )
    return value


def _point_key(point: Mapping[str, object]) -> tuple[object, ...]:
    metric = _mapping(point, "metric")
    target = _optional_mapping(point, "target")
    component = _optional_mapping(point, "component")
    pair = _optional_mapping(point, "pair")
    return (
        point.get("epoch"),
        _string(metric, "name"),
        None if target is None else target.get("identity"),
        None if target is None else target.get("index"),
        None if component is None else component.get("identity"),
        None if component is None else component.get("operator"),
        None if pair is None else pair.get("leftComponentIdentity"),
        None if pair is None else pair.get("rightComponentIdentity"),
    )


def _validate_health(health: JsonObject, path: str) -> None:
    completed = _integer(health, "trainingBatchesCompleted")
    applied = _integer(health, "optimizerUpdatesApplied")
    skipped = _integer(health, "optimizerUpdatesSkipped")
    overflow = _integer(health, "ampOverflowBatches")
    finite = _integer(health, "finiteGradientBatches")
    non_finite = _integer(health, "nonFiniteGradientBatches")
    if (
        completed <= 0
        or completed != applied + skipped
        or completed != finite + non_finite
        or overflow > skipped
        or overflow > non_finite
    ):
        _integrity(path, "training health counters are inconsistent")


def _exact_counter(value: float, epoch: int, field: str) -> int:
    if value < 0 or not value.is_integer() or value > 9_007_199_254_740_991:
        _integrity(f"/epochs/{epoch - 1}/health/{field}", "health counter is not an integer")
    return int(value)


def _metadata_mapping(model: PublishedModelRecord, field: str) -> Mapping[str, object]:
    return _mapping(cast(Mapping[str, object], model.metadata), field)


def _metadata_positive_integer(
    model: PublishedModelRecord,
    container: str,
    field: str,
) -> int:
    value = _integer(_metadata_mapping(model, container), field)
    if value < 1:
        raise TrainingTelemetryStoredMetadataError(
            f"/{container}/{field}",
            "stored metadata value must be positive",
        )
    return value


def _metadata_nonnegative_integer(
    model: PublishedModelRecord,
    container: str,
    field: str,
) -> int:
    value = _integer(_metadata_mapping(model, container), field)
    if value < 0:
        raise TrainingTelemetryStoredMetadataError(
            f"/{container}/{field}",
            "stored metadata value must be non-negative",
        )
    return value


def _mapping(document: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = document.get(field)
    if not isinstance(value, Mapping):
        _integrity(f"/{field}", "telemetry object is missing")
    return cast(Mapping[str, object], value)


def _optional_mapping(
    document: Mapping[str, object],
    field: str,
) -> Mapping[str, object] | None:
    value = document.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _integrity(f"/{field}", "telemetry context is invalid")
    return cast(Mapping[str, object], value)


def _sequence(document: Mapping[str, object], field: str) -> Sequence[object]:
    value = document.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _integrity(f"/{field}", "telemetry array is missing")
    return cast(Sequence[object], value)


def _string(document: Mapping[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str):
        _integrity(f"/{field}", "telemetry string is missing")
    return value


def _integer(document: Mapping[str, object], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        _integrity(f"/{field}", "telemetry integer is missing")
    return value


def _number(document: Mapping[str, object], field: str) -> float:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _integrity(f"/{field}", "telemetry number is missing")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _integrity(f"/{field}", "telemetry number is not finite")
    if not math.isfinite(result):
        _integrity(f"/{field}", "telemetry number is not finite")
    return result


def _optional_number(document: Mapping[str, object], field: str) -> float | None:
    value = document.get(field)
    if value is None:
        return None
    return _number(document, field)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _integrity(path: str, message: str) -> NoReturn:
    raise TrainingTelemetryIntegrityError(path, message)


__all__ = ["GetGradientInteractions", "GetTrainingTelemetryReport"]
