from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Protocol, cast

import rfc8785

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.metrics.fit_run.v11 import (
    RUN_INDEX,
    validate_run_document,
)
from app.contracts.metrics.v11 import POINT_INDEX, validate_point_document
from app.service.application.messages.training_telemetry import (
    ProjectedTrainingTelemetry,
)
from app.service.application.ports.telemetry import (
    BlockedMetricsDeliveryError,
    RetryableMetricsDeliveryError,
)
from app.service.application.ports.training_telemetry import (
    TrainingTelemetryBackendUnavailable,
    TrainingTelemetryIntegrityError,
)

_SEARCH_PAGE_SIZE = 1_000
_REPORT_METRICS = (
    "training.amp.overflow_batches",
    "training.batches.completed",
    "training.gradient.batches.finite",
    "training.gradient.batches.non_finite",
    "training.gradient.component.norm",
    "training.loss.auxiliary",
    "training.loss.direct",
    "training.loss.total",
    "training.optimizer.updates.applied",
    "training.optimizer.updates.skipped",
    "training.selection.score",
    "training.target.mae",
    "training.target.rmse",
)
_GRADIENT_METRICS = (
    "training.gradient.component.norm",
    "training.gradient.pair.cosine",
    "training.gradient.pair.negative_fraction",
)


class _DeliveryStatus(Protocol):
    def delivery_status(self, job_id: str) -> str | None: ...


class _TelemetryClient(Protocol):
    def document_source(
        self,
        index: str,
        document_id: str,
    ) -> JsonObject | None: ...

    def search_page(
        self,
        index: str,
        query: JsonObject,
    ) -> tuple[tuple[JsonObject, tuple[object, ...]], ...]: ...


class OpenSearchTrainingTelemetrySource:
    """Read the immutable metrics projection behind a neutral query port."""

    def __init__(
        self,
        client: _TelemetryClient,
        delivery_status: _DeliveryStatus,
        *,
        deployment_id: str,
        delivery_expected: Callable[[], bool],
    ) -> None:
        self._client = client
        self._delivery_status = delivery_status
        self._deployment_id = deployment_id
        self._delivery_expected = delivery_expected

    def load_report(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
    ) -> ProjectedTrainingTelemetry:
        try:
            summary = self._client.document_source(
                RUN_INDEX,
                _run_document_id(
                    self._deployment_id,
                    producing_run_id,
                    model_ref,
                ),
            )
            if summary is None:
                try:
                    status = self._delivery_status.delivery_status(
                        producing_run_id
                    )
                except RuntimeError as exc:
                    raise TrainingTelemetryBackendUnavailable from exc
                if status == "PENDING" and self._delivery_expected():
                    return ProjectedTrainingTelemetry(state="pending")
                return ProjectedTrainingTelemetry(
                    state="unavailable",
                    unavailable_reason="NO_COMPLETE_REPORT",
                )
            try:
                summary = validate_run_document(summary)
            except (OverflowError, TypeError, ValueError) as exc:
                raise TrainingTelemetryIntegrityError("", str(exc)) from exc
            if (
                summary.get("summaryId")
                != _run_document_id(
                    self._deployment_id,
                    producing_run_id,
                    model_ref,
                )
                or summary.get("deploymentId") != self._deployment_id
            ):
                raise TrainingTelemetryIntegrityError(
                    "",
                    "fit run completion marker identity differs",
                )
            points = self._points(
                model_ref=model_ref,
                producing_run_id=producing_run_id,
                metric_names=_REPORT_METRICS,
            )
            identity = _projection_identity(summary, points)
            return ProjectedTrainingTelemetry(
                state="available",
                report_identity=identity,
                run_document=summary,
                report_points=points,
            )
        except TrainingTelemetryIntegrityError:
            raise
        except (RetryableMetricsDeliveryError, BlockedMetricsDeliveryError) as exc:
            raise TrainingTelemetryBackendUnavailable from exc

    def load_gradient_points(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
        epoch: int,
    ) -> tuple[JsonObject, ...]:
        try:
            return self._points(
                model_ref=model_ref,
                producing_run_id=producing_run_id,
                metric_names=_GRADIENT_METRICS,
                epoch=epoch,
            )
        except TrainingTelemetryIntegrityError:
            raise
        except (RetryableMetricsDeliveryError, BlockedMetricsDeliveryError) as exc:
            raise TrainingTelemetryBackendUnavailable from exc

    def _points(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
        metric_names: Sequence[str],
        epoch: int | None = None,
    ) -> tuple[JsonObject, ...]:
        filters: list[JsonObject] = [
            {"term": {"schema": "transformer.metrics-point.v11"}},
            {"term": {"deploymentId": self._deployment_id}},
            {"term": {"runId": producing_run_id}},
            {"term": {"modelRef": model_ref}},
            {"terms": {"metric.name": list(metric_names)}},
        ]
        if epoch is not None:
            filters.append({"term": {"epoch": epoch}})
        result: list[JsonObject] = []
        after: tuple[int, str] | None = None
        while True:
            filters_value = cast(list[JsonValue], filters)
            query: JsonObject = {
                "size": _SEARCH_PAGE_SIZE,
                "track_total_hits": False,
                "query": {"bool": {"filter": filters_value}},
                "sort": [
                    {"epoch": {"order": "asc"}},
                    {"eventId": {"order": "asc"}},
                ],
            }
            if after is not None:
                query["search_after"] = list(cast(tuple[JsonValue, ...], after))
            page = self._client.search_page(POINT_INDEX, query)
            if not page:
                break
            for source, sort in page:
                try:
                    document = validate_point_document(source)
                    _verify_point_digest(document)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise TrainingTelemetryIntegrityError("", str(exc)) from exc
                metric = document.get("metric")
                if (
                    document.get("deploymentId") != self._deployment_id
                    or document.get("runId") != producing_run_id
                    or document.get("modelRef") != model_ref
                    or not isinstance(metric, dict)
                    or metric.get("name") not in metric_names
                    or (epoch is not None and document.get("epoch") != epoch)
                ):
                    raise TrainingTelemetryIntegrityError(
                        "",
                        "metrics point query identity differs",
                    )
                result.append(document)
                if (
                    len(sort) != 2
                    or isinstance(sort[0], bool)
                    or not isinstance(sort[0], int)
                    or not isinstance(sort[1], str)
                ):
                    raise TrainingTelemetryBackendUnavailable(
                        "OpenSearch sort boundary is invalid"
                    )
                boundary = cast(tuple[int, str], sort)
                if after is not None and boundary <= after:
                    raise TrainingTelemetryBackendUnavailable(
                        "OpenSearch sort boundary did not advance"
                    )
                after = boundary
            if len(page) < _SEARCH_PAGE_SIZE:
                break
        return tuple(result)


class UnavailableTrainingTelemetrySource:
    def load_report(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
    ) -> ProjectedTrainingTelemetry:
        del model_ref, producing_run_id
        raise TrainingTelemetryBackendUnavailable

    def load_gradient_points(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
        epoch: int,
    ) -> tuple[JsonObject, ...]:
        del model_ref, producing_run_id, epoch
        raise TrainingTelemetryBackendUnavailable


def _run_document_id(
    deployment_id: str,
    run_id: str,
    model_ref: str,
) -> str:
    return hashlib.sha256(
        rfc8785.dumps([
            "transformer.metrics-fit-run.v11",
            deployment_id,
            run_id,
            model_ref,
        ])
    ).hexdigest()


def _projection_identity(
    summary: JsonObject,
    points: Sequence[JsonObject],
) -> str:
    identities: list[JsonValue] = [
        cast(JsonValue, summary),
        *sorted(cast(str, point["documentSha256"]) for point in points),
    ]
    return hashlib.sha256(rfc8785.dumps(identities)).hexdigest()


def _verify_point_digest(document: JsonObject) -> None:
    expected = document.get("documentSha256")
    content = {
        key: value
        for key, value in document.items()
        if key != "documentSha256"
    }
    actual = hashlib.sha256(rfc8785.dumps(cast(JsonValue, content))).hexdigest()
    if expected != actual:
        raise ValueError("metrics point document digest differs")


__all__ = [
    "OpenSearchTrainingTelemetrySource",
    "UnavailableTrainingTelemetrySource",
]
