from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from app.contracts.json_types import JsonObject
from app.contracts.metrics.fit_run.v5 import (
    build_run_document,
    build_run_summary,
)
from app.contracts.metrics.v5 import (
    build_training_record,
    project_training_points,
)
from app.contracts.semantic.v1 import ModelContract
from app.contracts.training_telemetry.v1 import (
    validate_training_telemetry_document,
)
from app.project import PROJECT_ROOT
from app.service.adapters.inbound.flight.model_catalog import (
    CatalogModelMetadataVerifier,
)
from app.service.adapters.inbound.flight.training_telemetry import (
    present_gradient_interactions,
    present_training_telemetry_report,
)
from app.service.adapters.outbound.opensearch.training_telemetry import (
    OpenSearchTrainingTelemetrySource,
)
from app.service.application.messages.model_catalog import CatalogModelRecord
from app.service.application.messages.training_telemetry import (
    GetGradientInteractionsQuery,
    GetTrainingTelemetryReportQuery,
    ProjectedTrainingTelemetry,
)
from app.service.application.ports.model_catalog import CatalogModelNotFound
from app.service.application.ports.training_telemetry import (
    TrainingTelemetryIntegrityError,
)
from app.service.application.queries.training_telemetry import (
    GetGradientInteractions,
    GetTrainingTelemetryReport,
)
from app.service.application.services.training_telemetry_cursor import (
    ExpiredTrainingTelemetryCursor,
    InvalidTrainingTelemetryCursor,
)
from app.service.domain.errors import ServiceError
from app.service.domain.records import PublishedModelRecord

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ATTEMPT_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
DEPLOYMENT_ID = "test"
REPORT_METRICS = {
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
}
MODEL_CATALOG_FIXTURES = (
    PROJECT_ROOT / "app" / "contracts" / "model_catalog" / "v1" / "fixtures"
)
SEMANTIC_FIXTURES = (
    PROJECT_ROOT / "app" / "contracts" / "semantic" / "v1" / "fixtures"
)


class _Store:
    def __init__(self, model: PublishedModelRecord | None) -> None:
        self.model = model

    def cursor_signing_key(self) -> bytes:
        return b"training-telemetry-test-key".ljust(32, b"0")

    def get_model(
        self,
        owner_subject: str,
        model_ref: str,
    ) -> CatalogModelRecord | None:
        if (
            owner_subject != "consumer"
            or self.model is None
            or self.model.model_ref != model_ref
        ):
            return None
        return CatalogModelRecord(self.model, 1, NOW)


class _Source:
    def __init__(
        self,
        projection: ProjectedTrainingTelemetry,
        gradient_points: tuple[JsonObject, ...] = (),
    ) -> None:
        self.projection = projection
        self.gradient_points = gradient_points
        self.report_calls = 0

    def load_report(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
    ) -> ProjectedTrainingTelemetry:
        del model_ref, producing_run_id
        self.report_calls += 1
        return self.projection

    def load_gradient_points(
        self,
        *,
        model_ref: str,
        producing_run_id: str,
        epoch: int,
    ) -> tuple[JsonObject, ...]:
        del model_ref, producing_run_id, epoch
        return self.gradient_points


class _OpenSearchClient:
    def __init__(
        self,
        summary: JsonObject | None,
        points: tuple[JsonObject, ...] = (),
    ) -> None:
        self.summary = summary
        self.points = points
        self.queries: list[JsonObject] = []

    def document_source(self, index: str, document_id: str) -> JsonObject | None:
        assert index == "metrics-runs-v5"
        assert len(document_id) == 64
        return self.summary

    def search_page(
        self,
        index: str,
        query: JsonObject,
    ) -> tuple[tuple[JsonObject, tuple[object, ...]], ...]:
        assert index == "metrics-points-v5"
        self.queries.append(query)
        result = (
            (
                point,
                (
                    cast(int, point["epoch"]),
                    cast(str, point["eventId"]),
                ),
            )
            for point in self.points
        )
        return tuple(sorted(result, key=lambda item: item[1]))


class _DeliveryStatus:
    def __init__(self, status: str | None) -> None:
        self.status = status

    def delivery_status(self, job_id: str) -> str | None:
        assert job_id
        return self.status


def test_report_validates_complete_projection_and_pages_dense_epochs() -> None:
    model = _single_target_model(completed_epochs=3)
    source = _source_for_model(model)
    query = GetTrainingTelemetryReport(
        _Store(model),
        source,
        metadata_verifier=CatalogModelMetadataVerifier(),
        cursor_ttl_seconds=900,
        clock=lambda: NOW,
    )

    first = query.execute(GetTrainingTelemetryReportQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        page_size=2,
        cursor=None,
    ))
    first_document = present_training_telemetry_report(first)
    validate_training_telemetry_document(first_document, "report-result")

    assert first.state == "available"
    assert [item["epoch"] for item in first.epoch_items] == [1, 2]
    assert first.coverage == {
        "completedEpochs": 3,
        "firstEpoch": 1,
        "lastEpoch": 3,
    }
    assert first.health_totals == {
        "trainingBatchesCompleted": 6,
        "optimizerUpdatesApplied": 6,
        "optimizerUpdatesSkipped": 0,
        "ampOverflowBatches": 0,
        "finiteGradientBatches": 6,
        "nonFiniteGradientBatches": 0,
    }
    assert [(item["epoch"], item["roles"]) for item in first.anchors] == [
        (1, ["first"]),
        (3, ["published"]),
    ]
    assert first.gradient_interactions == {"state": "notConfigured"}
    assert first.next_cursor is not None
    assert first.cursor_expires_at == "2026-09-07T12:15:00.000000Z"

    second = query.execute(GetTrainingTelemetryReportQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        page_size=2,
        cursor=first.next_cursor,
    ))
    assert [item["epoch"] for item in second.epoch_items] == [3]
    assert second.next_cursor is None
    assert second.cursor_expires_at is None

    with pytest.raises(CatalogModelNotFound):
        query.execute(GetTrainingTelemetryReportQuery(
            owner_subject="other-consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
            page_size=2,
            cursor=first.next_cursor,
        ))
    with pytest.raises(InvalidTrainingTelemetryCursor):
        query.execute(GetTrainingTelemetryReportQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
            page_size=1,
            cursor=first.next_cursor,
        ))
    expired = GetTrainingTelemetryReport(
        _Store(model),
        source,
        metadata_verifier=CatalogModelMetadataVerifier(),
        cursor_ttl_seconds=900,
        clock=lambda: NOW + timedelta(seconds=900),
    )
    with pytest.raises(ExpiredTrainingTelemetryCursor):
        expired.execute(GetTrainingTelemetryReportQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
            page_size=2,
            cursor=first.next_cursor,
        ))


@pytest.mark.parametrize(
    ("projection", "expected_state", "expected_reason"),
    (
        (ProjectedTrainingTelemetry(state="pending"), "pending", None),
        (
            ProjectedTrainingTelemetry(
                state="unavailable",
                unavailable_reason="NO_COMPLETE_REPORT",
            ),
            "unavailable",
            "NO_COMPLETE_REPORT",
        ),
    ),
)
def test_report_preserves_non_error_availability_outcomes(
    projection: ProjectedTrainingTelemetry,
    expected_state: str,
    expected_reason: str | None,
) -> None:
    model = _single_target_model(completed_epochs=1)
    result = GetTrainingTelemetryReport(
        _Store(model),
        _Source(projection),
        metadata_verifier=CatalogModelMetadataVerifier(),
        cursor_ttl_seconds=900,
        clock=lambda: NOW,
    ).execute(GetTrainingTelemetryReportQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        page_size=1,
        cursor=None,
    ))

    document = present_training_telemetry_report(result)
    validate_training_telemetry_document(document, "report-result")
    assert result.state == expected_state
    assert result.unavailable_reason == expected_reason


def test_report_rejects_complete_projection_with_missing_epoch() -> None:
    model = _single_target_model(completed_epochs=2)
    source = _source_for_model(model)
    source.projection = ProjectedTrainingTelemetry(
        state="available",
        report_identity=source.projection.report_identity,
        run_document=source.projection.run_document,
        report_points=tuple(
            point
            for point in source.projection.report_points
            if point["epoch"] == 1
        ),
    )

    with pytest.raises(TrainingTelemetryIntegrityError, match="epoch 2"):
        GetTrainingTelemetryReport(
            _Store(model),
            source,
            metadata_verifier=CatalogModelMetadataVerifier(),
            cursor_ttl_seconds=900,
            clock=lambda: NOW,
        ).execute(GetTrainingTelemetryReportQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
            page_size=2,
            cursor=None,
        ))


def test_model_metadata_is_validated_before_telemetry_backend_access() -> None:
    model = _single_target_model(completed_epochs=1)
    model.metadata["jobId"] = "not-a-uuid"
    source = _Source(ProjectedTrainingTelemetry(state="pending"))

    with pytest.raises(ServiceError) as error:
        GetTrainingTelemetryReport(
            _Store(model),
            source,
            metadata_verifier=CatalogModelMetadataVerifier(),
            cursor_ttl_seconds=900,
            clock=lambda: NOW,
        ).execute(GetTrainingTelemetryReportQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
            page_size=1,
            cursor=None,
        ))

    assert error.value.detail is not None
    assert error.value.detail["reason"] == "STORED_MODEL_METADATA_INVALID"
    assert source.report_calls == 0


def test_model_deletion_wins_over_report_cursor_continuation() -> None:
    model = _single_target_model(completed_epochs=2)
    store = _Store(model)
    query = GetTrainingTelemetryReport(
        store,
        _source_for_model(model),
        metadata_verifier=CatalogModelMetadataVerifier(),
        cursor_ttl_seconds=900,
        clock=lambda: NOW,
    )
    first = query.execute(GetTrainingTelemetryReportQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        page_size=1,
        cursor=None,
    ))
    assert first.next_cursor is not None
    store.model = None

    with pytest.raises(CatalogModelNotFound):
        query.execute(GetTrainingTelemetryReportQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
            page_size=1,
            cursor=first.next_cursor,
        ))


def test_gradient_query_returns_objective_order_and_sparse_pair_pages() -> None:
    model = _multi_target_model()
    source = _source_for_model(model, gradient_epochs=frozenset({2}))
    report = GetTrainingTelemetryReport(
        _Store(model),
        source,
        metadata_verifier=CatalogModelMetadataVerifier(),
        cursor_ttl_seconds=900,
        clock=lambda: NOW,
    ).execute(GetTrainingTelemetryReportQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        page_size=2,
        cursor=None,
    ))
    assert report.gradient_interactions == {
        "state": "available",
        "collectedEpochCount": 1,
        "defaultEpoch": 2,
        "publishedEpochCollected": True,
    }

    query = GetGradientInteractions(
        _Store(model),
        source,
        metadata_verifier=CatalogModelMetadataVerifier(),
        cursor_ttl_seconds=900,
        clock=lambda: NOW,
    )
    first = query.execute(GetGradientInteractionsQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        epoch=2,
        page_size=1,
        cursor=None,
    ))
    first_document = present_gradient_interactions(first)
    validate_training_telemetry_document(
        first_document,
        "gradient-interactions-result",
    )
    contract = ModelContract.from_document(model.model_contract)
    expected_components = [
        cast(str, component["identity"])
        for component in (
            *contract.direct_components,
            *contract.auxiliary_components,
        )
    ]
    assert [item["componentIdentity"] for item in first.components] == (
        expected_components
    )
    assert len(first.pair_items) == 1
    assert first.next_cursor is not None

    second = query.execute(GetGradientInteractionsQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        model_ref=model.model_ref,
        epoch=2,
        page_size=1,
        cursor=first.next_cursor,
    ))
    assert len(second.pair_items) == 1
    assert second.next_cursor is None
    pairs = (*first.pair_items, *second.pair_items)
    identities = [
        (
            item["leftComponentIdentity"],
            item["rightComponentIdentity"],
        )
        for item in pairs
    ]
    assert identities == sorted(identities)
    assert len(identities) == 2


def test_opensearch_source_reads_completion_marker_and_verified_points() -> None:
    model = _single_target_model(completed_epochs=1)
    fixture = _source_for_model(model).projection
    client = _OpenSearchClient(
        cast(JsonObject, fixture.run_document),
        fixture.report_points,
    )

    projected = OpenSearchTrainingTelemetrySource(
        client,
        _DeliveryStatus("DELIVERED"),
        deployment_id=DEPLOYMENT_ID,
    ).load_report(
        model_ref=model.model_ref,
        producing_run_id=cast(str, model.metadata["jobId"]),
    )

    assert projected.state == "available"
    assert {point["eventId"] for point in projected.report_points} == {
        point["eventId"] for point in fixture.report_points
    }
    assert projected.report_identity is not None
    assert len(projected.report_identity) == 64
    assert client.queries


@pytest.mark.parametrize(
    ("delivery_status", "expected_state"),
    (("PENDING", "pending"), (None, "unavailable")),
)
def test_opensearch_source_classifies_missing_completion_marker(
    delivery_status: str | None,
    expected_state: str,
) -> None:
    model = _single_target_model(completed_epochs=1)
    projected = OpenSearchTrainingTelemetrySource(
        _OpenSearchClient(None),
        _DeliveryStatus(delivery_status),
        deployment_id=DEPLOYMENT_ID,
    ).load_report(
        model_ref=model.model_ref,
        producing_run_id=cast(str, model.metadata["jobId"]),
    )

    assert projected.state == expected_state


def test_opensearch_source_rejects_point_with_changed_content() -> None:
    model = _single_target_model(completed_epochs=1)
    fixture = _source_for_model(model).projection
    point = dict(fixture.report_points[0])
    point["metric"] = {
        **cast(JsonObject, point["metric"]),
        "value": 123.0,
    }
    source = OpenSearchTrainingTelemetrySource(
        _OpenSearchClient(
            cast(JsonObject, fixture.run_document),
            (point,),
        ),
        _DeliveryStatus("DELIVERED"),
        deployment_id=DEPLOYMENT_ID,
    )

    with pytest.raises(TrainingTelemetryIntegrityError, match="digest"):
        source.load_report(
            model_ref=model.model_ref,
            producing_run_id=cast(str, model.metadata["jobId"]),
        )


def _single_target_model(*, completed_epochs: int) -> PublishedModelRecord:
    result = json.loads(
        (MODEL_CATALOG_FIXTURES / "detail.result.random.json").read_text(
            encoding="utf-8"
        )
    )
    detail = result["model"]
    detail["progress"]["completedEpochs"] = completed_epochs
    detail["progress"]["globalStep"] = completed_epochs * 10
    return _published_model(detail)


def _multi_target_model() -> PublishedModelRecord:
    fixture = json.loads(
        (SEMANTIC_FIXTURES / "multi-target-shared-resource.json").read_text(
            encoding="utf-8"
        )
    )
    contract = ModelContract.from_document(fixture["modelContract"])
    data_contract: JsonObject = {
        "identity": "consumer.learning-dataset",
        "revision": 1,
        "profile": "consumer.profile.multi-target",
        "dataContractSha256": "a" * 64,
        "seqLen": 3,
        "featureDim": 12,
    }
    digests = contract.digests("a" * 64)
    model_ref = "mdl_33333333333333333333333333333333"
    detail: JsonObject = {
        "summary": {
            "modelRef": model_ref,
            "label": "consumer-model",
            "generation": 1,
            "createdAt": "2026-09-07T12:00:00.000000Z",
            "semanticDigests": digests,
            "modelConfig": contract.model_config,
            "targetIdentities": list(contract.target_identities),
            "initialization": {"kind": "random"},
            "producingRunId": "33333333-3333-4333-8333-333333333333",
            "checkpoint": {
                "format": "transformer-checkpoint-v6",
                "sha256": "d" * 64,
                "bytes": 1024,
            },
        },
        "dataContract": data_contract,
        "modelContract": contract.to_document(),
        "trainingConfig": {
            "lr": 0.0005,
            "batchSize": 8,
            "epochs": 2,
            "useAmp": False,
            "weightDecay": 0.00001,
            "selection": None,
            "seed": 42,
            "deterministic": True,
        },
        "diagnostics": {
            "schemaVersion": 1,
            "gradientInteractions": {"everySteps": 10, "maxBatches": 1},
        },
        "selection": {
            "enabled": False,
            "modelContractSha256": digests["modelContractSha256"],
            "bestSelectionScore": None,
            "bestEpoch": None,
            "source": "last_epoch",
        },
        "progress": {
            "completedEpochs": 2,
            "globalStep": 20,
            "trainingComplete": True,
        },
        "initialization": {"kind": "random"},
        "jobConfigSha256": "7" * 64,
    }
    return _published_model(detail)


def _published_model(detail: JsonObject) -> PublishedModelRecord:
    summary = cast(JsonObject, detail["summary"])
    checkpoint = cast(JsonObject, summary["checkpoint"])
    metadata: JsonObject = {
        "format": "transformer-checkpoint-v6",
        "serviceVersion": "0.2.0",
        "generation": cast(JsonObject, detail["progress"])["completedEpochs"],
        "jobId": summary["producingRunId"],
        "dataContract": detail["dataContract"],
        "modelContract": detail["modelContract"],
        "semanticDigests": summary["semanticDigests"],
        "trainingConfig": detail["trainingConfig"],
        "diagnostics": detail["diagnostics"],
        "selection": detail["selection"],
        "initialization": detail["initialization"],
        "jobConfigSha256": detail["jobConfigSha256"],
        "manifestSha256": "f" * 64,
        "progress": detail["progress"],
        "modelRef": summary["modelRef"],
        "label": summary["label"],
        "checkpoint": checkpoint,
    }
    created_at = datetime.fromisoformat(cast(str, summary["createdAt"])).timestamp()
    return PublishedModelRecord(
        model_ref=cast(str, summary["modelRef"]),
        owner_subject="consumer",
        label=cast(str, summary["label"]),
        generation=cast(int, summary["generation"]),
        checkpoint_path=(
            f"models/{summary['modelRef']}/checkpoint.pth"
        ),
        byte_count=cast(int, checkpoint["bytes"]),
        sha256=cast(str, checkpoint["sha256"]),
        metadata=metadata,
        data_contract=cast(JsonObject, detail["dataContract"]),
        model_contract=cast(JsonObject, detail["modelContract"]),
        semantic_digests=cast(JsonObject, summary["semanticDigests"]),
        producing_job_id=cast(str, summary["producingRunId"]),
        created_at=created_at,
    )


def _source_for_model(
    model: PublishedModelRecord,
    *,
    gradient_epochs: frozenset[int] = frozenset(),
) -> _Source:
    completed_epochs = cast(int, cast(JsonObject, model.metadata["progress"])[
        "completedEpochs"
    ])
    all_points: list[JsonObject] = []
    for epoch in range(1, completed_epochs + 1):
        record = _training_record(
            model,
            epoch=epoch,
            step=epoch * 10,
            gradient_interactions=epoch in gradient_epochs,
        )
        all_points.extend(project_training_points(
            record,
            deployment_id=DEPLOYMENT_ID,
        ))
    report_points = tuple(
        point
        for point in all_points
        if cast(JsonObject, point["metric"])["name"] in REPORT_METRICS
    )
    gradient_points = tuple(
        point
        for point in all_points
        if point["epoch"] in gradient_epochs
        and cast(JsonObject, point["metric"])["name"]
        in {
            "training.gradient.component.norm",
            "training.gradient.pair.cosine",
            "training.gradient.pair.negative_fraction",
        }
    )
    run = _run_document(model)
    projection = ProjectedTrainingTelemetry(
        state="available",
        report_identity="a" * 64,
        run_document=run,
        report_points=report_points,
    )
    return _Source(projection, gradient_points)


def _training_record(
    model: PublishedModelRecord,
    *,
    epoch: int,
    step: int,
    gradient_interactions: bool,
) -> JsonObject:
    contract = ModelContract.from_document(model.model_contract)
    direct = [
        {
            "componentIdentity": component["identity"],
            "operator": component["operator"],
            "targetIdentity": contract.target_identities[index],
            "targetIndex": index,
            "value": 0.1 * (index + 1) + epoch / 1000,
        }
        for index, component in enumerate(contract.direct_components)
    ]
    auxiliary = [
        {
            "componentIdentity": component["identity"],
            "operator": component["operator"],
            "value": -0.01 * (index + 1),
        }
        for index, component in enumerate(contract.auxiliary_components)
    ]
    targets = [
        {
            "targetIdentity": identity,
            "targetIndex": index,
            "mae": 0.01 * (index + 1),
            "rmse": 0.02 * (index + 1),
        }
        for index, identity in enumerate(contract.target_identities)
    ]
    gradient = None
    if gradient_interactions:
        components = [
            *contract.direct_components,
            *contract.auxiliary_components,
        ]
        gradient_components: list[JsonObject] = []
        for index, component in enumerate(components):
            item: JsonObject = {
                "componentIdentity": component["identity"],
                "meanNorm": 0.5 + index,
            }
            if index < len(contract.target_identities):
                item["targetIdentity"] = contract.target_identities[index]
                item["targetIndex"] = index
            gradient_components.append(item)
        gradient = {
            "samples": 1,
            "components": gradient_components,
            "pairs": [
                {
                    "leftComponentIdentity": components[0]["identity"],
                    "rightComponentIdentity": components[3]["identity"],
                    "meanCosine": -0.25,
                    "negativeCosineFraction": 1.0,
                },
                {
                    "leftComponentIdentity": components[1]["identity"],
                    "rightComponentIdentity": components[4]["identity"],
                    "meanCosine": 0.5,
                    "negativeCosineFraction": 0.0,
                },
            ],
        }
    metrics: JsonObject = {
        "mode": "fit-stream",
        "frame": None,
        "epoch": epoch,
        "step": step,
        "rows": 4,
        "batches": 2,
        "lr": 0.001,
        "loss": 0.25 + epoch / 1000,
        "directLosses": direct,
        "auxiliaryLosses": auxiliary,
        "targetMetrics": targets,
        "gradientInteractions": gradient,
        "selectionScore": None,
        "trainingBatchesCompleted": 2,
        "optimizerUpdatesApplied": 2,
        "optimizerUpdatesSkipped": 0,
        "ampOverflowBatches": 0,
        "finiteGradientBatches": 2,
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
    return build_training_record(
        metrics,
        recorded_at=NOW.timestamp(),
        job_id=cast(str, model.metadata["jobId"]),
        attempt_id=ATTEMPT_ID,
        attempt=1,
        model_ref=model.model_ref,
        semantic_digests=model.semantic_digests,
        checkpoint_format="transformer-checkpoint-v6",
        application_version="0.2.0",
        git_commit="0" * 40,
        targets=contract.target_identities,
    )


def _run_document(model: PublishedModelRecord) -> JsonObject:
    run_id = cast(str, model.metadata["jobId"])
    contract = ModelContract.from_document(model.model_contract)
    summary = build_run_summary(
        recorded_at=NOW.timestamp(),
        job_id=run_id,
        attempt_id=ATTEMPT_ID,
        attempt=1,
        model_ref=model.model_ref,
        semantic_digests=model.semantic_digests,
        job_config_sha256=cast(str, model.metadata["jobConfigSha256"]),
        checkpoint_format="transformer-checkpoint-v6",
        application_version="0.2.0",
        git_commit="0" * 40,
        targets=contract.target_identities,
        initialization=cast(JsonObject, model.metadata["initialization"]),
        milestones={
            "createdAt": "2026-09-07T11:00:00.000Z",
            "firstInputCommittedAt": "2026-09-07T11:01:00.000Z",
            "inputClosedAt": "2026-09-07T11:02:00.000Z",
            "workerCompletedAt": "2026-09-07T11:58:00.000Z",
            "publishedAt": "2026-09-07T12:00:00.000Z",
        },
        durations={
            "firstInputWaitMs": 60_000.0,
            "eofWaitMs": 60_000.0,
            "queueWaitMs": 0.0,
            "workerStartupMs": 1_000.0,
            "trainingMs": 3_300_000.0,
            "checkpointSerializationMs": 100.0,
            "checkpointPublicationMs": 100.0,
            "modelPublicationMs": 120_000.0,
            "remoteFitMs": 3_600_000.0,
        },
        counts={
            "attempts": 1,
            "recoveries": 0,
            "inputPayloads": 1,
            "inputChunks": 1,
            "logicalRows": 4,
            "nativeRows": [4],
            "inputBytes": 16,
        },
    )
    return build_run_document(
        summary,
        deployment_id=DEPLOYMENT_ID,
        byte_count=1,
        sha256="e" * 64,
    )
