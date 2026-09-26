import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.contracts.flight.v22.codec import validate_request_document
from app.contracts.flight.v22.constants import ACTIONS
from app.contracts.semantic.v5 import ModelContract
from app.contracts.target_head_diagnostics.v5.constants import REPORT_ACTION
from app.contracts.worker.v20.config import ModelConfig
from app.contracts.worker.v20.model_definition import resolved_semantic_digests
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.adapters.inbound.flight.validation import (
    validate_action_request,
)
from app.service.application.messages.model_catalog import CatalogModelRecord
from app.service.application.messages.target_head_diagnostics import (
    GetTargetHeadDiagnosticsReportQuery,
)
from app.service.application.ports.model_catalog import CatalogModelNotFound
from app.service.application.queries.target_head_diagnostics import (
    GetTargetHeadDiagnosticsReport,
)
from app.service.application.services.training_telemetry_snapshot import (
    TrainingTelemetrySnapshotStore,
)
from app.service.domain.records import PublishedModelRecord
from tests.fixture_documents import semantic_fixture_document


def _model(
    *,
    target_head: str | None,
    artifact_format: str = "transformer-target-head-diagnostics-v5",
) -> PublishedModelRecord:
    fixture = semantic_fixture_document("positive-class-weighted-binary-w28")
    model_contract = fixture["modelContract"]
    assert isinstance(model_contract, dict)
    contract = ModelContract.from_document(model_contract)
    model_config = ModelConfig.from_tuning(
        contract.model_tuning,
        seq_len=2,
        feature_dim=8,
    )
    semantic_digests = resolved_semantic_digests(
        contract,
        "d" * 64,
        model_config,
    )
    metadata = {
        "jobId": "22222222-2222-4222-8222-222222222222",
        "jobConfigSha256": "c" * 64,
        "manifestSha256": "a" * 64,
        "progress": {"completedEpochs": 2},
        "diagnostics": {
            "schemaVersion": 3,
            "gradientInteractions": None,
            "targetHead": target_head,
            "encoderLayerDiagnostics": None,
        },
    }
    if target_head is not None:
        metadata["targetHeadDiagnostics"] = _artifact(
            str(semantic_digests["modelDefinitionSha256"]),
            artifact_format,
        )
    return PublishedModelRecord(
        model_ref="mdl_11111111111111111111111111111111",
        owner_subject="owner-a",
        label="diagnostics",
        generation=1,
        checkpoint_path="/missing/checkpoint.pth",
        byte_count=1,
        sha256="b" * 64,
        metadata=metadata,
        data_contract={"seqLen": 2, "featureDim": 8},
        model_contract=contract.to_document(),
        semantic_digests=semantic_digests,
        producing_job_id="22222222-2222-4222-8222-222222222222",
        created_at=0,
    )


def _artifact(
    model_definition_sha256: str,
    artifact_format: str,
) -> dict[str, object]:
    return {
        "format": artifact_format,
        "jobId": "22222222-2222-4222-8222-222222222222",
        "attempt": 1,
        "attemptId": "33333333-3333-4333-8333-333333333333",
        "inputRevision": 1,
        "manifestSha256": "a" * 64,
        "modelDefinitionSha256": model_definition_sha256,
        "jobConfigSha256": "c" * 64,
        "sampleIdentity": "thds_" + "e" * 64,
        "sampleRowCount": 4,
        "layout": [
            {
                "targetIdentity": "OpaqueBinaryEvent",
                "targetIndex": 0,
                "directComponentIdentity": "direct.opaque-binary-event",
            },
        ],
        "coverage": {"completedEpochs": 2},
        "epochs": [_epoch(1, 1), _epoch(2, 2)],
    }


def _epoch(epoch: int, global_step: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "globalStep": global_step,
        "representationFlow": {
            "encoderInput": {"rowCenteredL2Mean": 0.7},
            "encoderLayers": [
                {"layerIndex": 0, "rowCenteredL2Mean": 0.6},
            ],
            "targetHeadInput": {"rowCenteredL2Mean": 0.5},
        },
        "encoderBlockFlow": {
            "normalizationOrder": "postNorm",
            "layers": [
                {
                    "layerIndex": 0,
                    "input": {"rowCenteredL2Mean": 0.7},
                    "attentionResidual": {"rowCenteredL2Mean": 0.65},
                    "norm1": {"rowCenteredL2Mean": 0.6},
                    "feedForwardResidual": {"rowCenteredL2Mean": 0.62},
                    "norm2": {"rowCenteredL2Mean": 0.6},
                },
            ],
            "outputHeadShared": {
                "linear": {"rowCenteredL2Mean": 0.55},
                "gelu": {"rowCenteredL2Mean": 0.52},
                "layerNorm": {"rowCenteredL2Mean": 0.5},
            },
        },
        "targetHeads": [
            {
                "rawLogit": _distribution(),
                "publicPrediction": _distribution(),
                "gradientL2Mean": 0.1,
                "gradientL2Maximum": 0.2,
                "gradientBatchCount": 2,
                "weightL2AfterEpoch": 0.3,
                "biasAfterEpoch": 0.4,
            },
        ],
    }


def _distribution() -> dict[str, object]:
    return {
        "rowCount": 4,
        "minimum": 0.0,
        "maximum": 1.0,
        "mean": 0.5,
        "standardDeviation": 0.25,
    }


def _query(model: PublishedModelRecord) -> GetTargetHeadDiagnosticsReport:
    entry = CatalogModelRecord(
        model=model,
        catalog_ordinal=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
    )
    store = SimpleNamespace(
        get_model=lambda owner, model_ref: (
            entry
            if (owner, model_ref) == ("owner-a", model.model_ref)
            else None
        ),
        cursor_signing_key=lambda: b"k" * 32,
    )
    snapshots = TrainingTelemetrySnapshotStore(
        max_count=64,
        max_total_bytes=64 * 1024 * 1024,
        max_snapshot_bytes=16 * 1024 * 1024,
        boot_identity="f" * 32,
    )
    return GetTargetHeadDiagnosticsReport(
        store,
        snapshots,
        metadata_verifier=SimpleNamespace(verify=lambda _model: None),
        cursor_ttl_seconds=900,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_target_head_diagnostics_returns_retained_epoch_pages():
    model = _model(target_head="fullCommittedArtifact")
    query = _query(model)
    initial = query.execute(
        GetTargetHeadDiagnosticsReportQuery(
            owner_subject="owner-a",
            request_id="11111111-1111-4111-8111-111111111111",
            model_ref=model.model_ref,
            page_size=1,
            cursor=None,
        )
    )

    assert initial["state"] == "available"
    assert initial["epochPage"]["items"][0]["epoch"] == 1
    cursor = initial["epochPage"]["nextCursor"]
    assert isinstance(cursor, str)

    continuation = query.execute(
        GetTargetHeadDiagnosticsReportQuery(
            owner_subject="owner-a",
            request_id="44444444-4444-4444-8444-444444444444",
            model_ref=model.model_ref,
            page_size=1,
            cursor=cursor,
        )
    )

    assert continuation["state"] == "available"
    assert continuation["epochPage"]["items"][0]["epoch"] == 2
    assert continuation["epochPage"]["nextCursor"] is None


def test_target_head_diagnostics_reports_not_configured_without_opt_in():
    model = _model(target_head=None)

    result = _query(model).execute(
        GetTargetHeadDiagnosticsReportQuery(
            owner_subject="owner-a",
            request_id="11111111-1111-4111-8111-111111111111",
            model_ref=model.model_ref,
            page_size=1,
            cursor=None,
        )
    )

    assert result == {
        "requestId": "11111111-1111-4111-8111-111111111111",
        "state": "notConfigured",
        "modelRef": model.model_ref,
        "reason": "TARGET_HEAD_DIAGNOSTICS_NOT_CONFIGURED",
    }


def test_target_head_diagnostics_reports_unsupported_artifact_format():
    model = _model(
        target_head="fullCommittedArtifact",
        artifact_format="transformer-target-head-diagnostics-v1",
    )

    result = _query(model).execute(
        GetTargetHeadDiagnosticsReportQuery(
            owner_subject="owner-a",
            request_id="11111111-1111-4111-8111-111111111111",
            model_ref=model.model_ref,
            page_size=1,
            cursor=None,
        )
    )

    assert result == {
        "requestId": "11111111-1111-4111-8111-111111111111",
        "state": "unavailable",
        "modelRef": model.model_ref,
        "reason": "FORMAT_UNSUPPORTED",
    }


def test_target_head_diagnostics_rejects_wrong_encoder_layer_order():
    model = _model(target_head="fullCommittedArtifact")
    artifact = model.metadata["targetHeadDiagnostics"]
    assert isinstance(artifact, dict)
    epochs = artifact["epochs"]
    assert isinstance(epochs, list)
    first_epoch = epochs[0]
    assert isinstance(first_epoch, dict)
    flow = first_epoch["representationFlow"]
    assert isinstance(flow, dict)
    layers = flow["encoderLayers"]
    assert isinstance(layers, list)
    first_layer = layers[0]
    assert isinstance(first_layer, dict)
    first_layer["layerIndex"] = 1

    with pytest.raises(ValueError, match="encoder layer order"):
        _query(model).execute(
            GetTargetHeadDiagnosticsReportQuery(
                owner_subject="owner-a",
                request_id="11111111-1111-4111-8111-111111111111",
                model_ref=model.model_ref,
                page_size=1,
                cursor=None,
            )
        )


def test_target_head_diagnostics_rejects_wrong_encoder_block_layer_order():
    model = _model(target_head="fullCommittedArtifact")
    artifact = model.metadata["targetHeadDiagnostics"]
    assert isinstance(artifact, dict)
    epochs = artifact["epochs"]
    assert isinstance(epochs, list)
    first_epoch = epochs[0]
    assert isinstance(first_epoch, dict)
    block_flow = first_epoch["encoderBlockFlow"]
    assert isinstance(block_flow, dict)
    layers = block_flow["layers"]
    assert isinstance(layers, list)
    first_layer = layers[0]
    assert isinstance(first_layer, dict)
    first_layer["layerIndex"] = 1

    with pytest.raises(ValueError, match="encoder block layer order"):
        _query(model).execute(
            GetTargetHeadDiagnosticsReportQuery(
                owner_subject="owner-a",
                request_id="11111111-1111-4111-8111-111111111111",
                model_ref=model.model_ref,
                page_size=1,
                cursor=None,
            )
        )


def test_target_head_diagnostics_rejects_wrong_encoder_normalization_order():
    model = _model(target_head="fullCommittedArtifact")
    artifact = model.metadata["targetHeadDiagnostics"]
    assert isinstance(artifact, dict)
    epochs = artifact["epochs"]
    assert isinstance(epochs, list)
    first_epoch = epochs[0]
    assert isinstance(first_epoch, dict)
    block_flow = first_epoch["encoderBlockFlow"]
    assert isinstance(block_flow, dict)
    block_flow["normalizationOrder"] = "preNorm"

    with pytest.raises(ValueError, match="normalization order"):
        _query(model).execute(
            GetTargetHeadDiagnosticsReportQuery(
                owner_subject="owner-a",
                request_id="11111111-1111-4111-8111-111111111111",
                model_ref=model.model_ref,
                page_size=1,
                cursor=None,
            )
        )


def test_target_head_diagnostics_hides_model_deleted_while_report_is_built():
    model = _model(target_head="fullCommittedArtifact")
    entry = CatalogModelRecord(
        model=model,
        catalog_ordinal=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
    )
    calls = 0

    def get_model(owner: str, model_ref: str):
        nonlocal calls
        calls += 1
        if calls == 1 and (owner, model_ref) == ("owner-a", model.model_ref):
            return entry
        return None

    store = SimpleNamespace(
        get_model=get_model,
        cursor_signing_key=lambda: b"k" * 32,
    )
    snapshots = TrainingTelemetrySnapshotStore(
        max_count=64,
        max_total_bytes=64 * 1024 * 1024,
        max_snapshot_bytes=16 * 1024 * 1024,
        boot_identity="f" * 32,
    )
    query = GetTargetHeadDiagnosticsReport(
        store,
        snapshots,
        metadata_verifier=SimpleNamespace(verify=lambda _model: None),
        cursor_ttl_seconds=900,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    with pytest.raises(CatalogModelNotFound):
        query.execute(
            GetTargetHeadDiagnosticsReportQuery(
                owner_subject="owner-a",
                request_id="11111111-1111-4111-8111-111111111111",
                model_ref=model.model_ref,
                page_size=100,
                cursor=None,
            )
        )


def test_flight_v22_dispatches_target_head_diagnostics_report():
    model = _model(target_head="fullCommittedArtifact")
    query = _query(model)
    coordinator = JobCoordinator(
        create_job=SimpleNamespace(),
        acquire_job=SimpleNamespace(),
        close_input=SimpleNamespace(),
        cancel_job=SimpleNamespace(),
        get_status=SimpleNamespace(),
        list_inputs=SimpleNamespace(),
        list_outputs=SimpleNamespace(),
        list_catalog_models=SimpleNamespace(),
        get_catalog_model=SimpleNamespace(),
        get_model_topology=SimpleNamespace(),
        service_status=SimpleNamespace(),
        availability=SimpleNamespace(),
        get_target_head_diagnostics_report=query,
    )
    document = {
        "requestId": "11111111-1111-4111-8111-111111111111",
        "modelRef": model.model_ref,
        "pageSize": 1,
        "cursor": None,
    }

    assert REPORT_ACTION in ACTIONS
    request = validate_action_request(REPORT_ACTION, document)
    response = json.loads(
        coordinator.dispatch(REPORT_ACTION, "owner-a", request, document)
    )

    assert response["state"] == "available"
    assert response["modelRef"] == model.model_ref


def test_flight_v22_capabilities_advertise_target_head_normalization_orders():
    coordinator = JobCoordinator(
        create_job=SimpleNamespace(),
        acquire_job=SimpleNamespace(),
        close_input=SimpleNamespace(),
        cancel_job=SimpleNamespace(),
        get_status=SimpleNamespace(),
        list_inputs=SimpleNamespace(),
        list_outputs=SimpleNamespace(),
        list_catalog_models=SimpleNamespace(),
        get_catalog_model=SimpleNamespace(),
        get_model_topology=SimpleNamespace(),
        service_status=SimpleNamespace(
            capabilities=lambda: SimpleNamespace(
                device_inventory=SimpleNamespace(cuda_capacity=1),
                limits=SimpleNamespace(
                    max_payload_bytes=536870912,
                    max_rows_per_payload=2000000,
                    max_payloads_per_job=100000,
                    max_job_bytes=68719476736,
                    input_idle_timeout_seconds=900,
                ),
            )
        ),
        availability=SimpleNamespace(),
        get_target_head_diagnostics_report=SimpleNamespace(),
    )

    document = coordinator.capabilities(
        "11111111-1111-4111-8111-111111111111"
    )
    validate_request_document(document, "capabilities-result")

    diagnostics = document["queries"]["targetHeadDiagnostics"]
    assert diagnostics["revision"] == 5
    assert diagnostics["actions"] == [REPORT_ACTION]
    assert diagnostics["encoderLayerDiagnosticsModes"] == [
        "directComponentPerBatch",
    ]
    assert diagnostics["encoderNormalizationOrders"] == ["postNorm", "preNorm"]
    assert document["semantic"]["objectiveLanguage"]["revision"] == 5
    assert document["semantic"]["encoderNormalizationOrders"] == [
        "postNorm",
        "preNorm",
    ]
