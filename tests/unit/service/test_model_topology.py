import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.contracts.flight.v18.codec import validate_request_document
from app.contracts.flight.v18.constants import ACTIONS
from app.contracts.model_topology.v2 import validate_model_topology_document
from app.contracts.model_topology.v2.constants import DETAIL_ACTION
from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v16.config import ModelConfig
from app.contracts.worker.v16.model_definition import resolved_semantic_digests
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.adapters.inbound.flight.model_topology import (
    model_topology_response,
    present_model_topology,
)
from app.service.adapters.inbound.flight.validation import (
    validate_action_request,
)
from app.service.application.messages.model_catalog import CatalogModelRecord
from app.service.application.messages.model_topology import GetModelTopologyQuery
from app.service.application.ports.model_catalog import CatalogModelNotFound
from app.service.application.queries.model_topology import GetModelTopology
from app.service.application.services.model_topology import ModelTopologyBuilder
from app.service.domain.records import PublishedModelRecord
from tests.fixture_documents import semantic_fixture_document


def _model() -> PublishedModelRecord:
    document = semantic_fixture_document("multi-target-shared-resource")
    contract = ModelContract.from_document(document["modelContract"])
    config = ModelConfig.from_tuning(
        contract.model_tuning,
        seq_len=3,
        feature_dim=12,
    )
    return PublishedModelRecord(
        model_ref="mdl_11111111111111111111111111111111",
        owner_subject="owner-a",
        label="topology",
        generation=1,
        checkpoint_path="/missing/checkpoint.pth",
        byte_count=1,
        sha256="b" * 64,
        metadata={"modelConfig": config.to_manifest()},
        data_contract={"seqLen": 3, "featureDim": 12},
        model_contract=contract.to_document(),
        semantic_digests=resolved_semantic_digests(contract, "a" * 64, config),
        producing_job_id="22222222-2222-4222-8222-222222222222",
        created_at=0,
    )


def _weighted_binary_model() -> PublishedModelRecord:
    document = semantic_fixture_document("positive-class-weighted-binary-w28")
    contract = ModelContract.from_document(document["modelContract"])
    config = ModelConfig.from_tuning(
        contract.model_tuning,
        seq_len=2,
        feature_dim=8,
    )
    return PublishedModelRecord(
        model_ref="mdl_33333333333333333333333333333333",
        owner_subject="owner-a",
        label="topology",
        generation=1,
        checkpoint_path="/missing/checkpoint.pth",
        byte_count=1,
        sha256="b" * 64,
        metadata={"modelConfig": config.to_manifest()},
        data_contract={"seqLen": 2, "featureDim": 8},
        model_contract=contract.to_document(),
        semantic_digests=resolved_semantic_digests(contract, "a" * 64, config),
        producing_job_id="22222222-2222-4222-8222-222222222222",
        created_at=0,
    )


def test_topology_materializes_public_execution_graph():
    model = _model()

    topology = ModelTopologyBuilder().build(model)

    nodes = {node["id"]: node for node in topology["nodes"]}
    assert len(nodes) == 27
    assert len(topology["edges"]) == 38
    assert nodes["encoder-1"]["attentionHeadCount"] == 4
    assert nodes["encoder-2"]["encoderLayer"] == 2
    assert nodes["target-0-loss-input"]["transformation"] == "Identity"
    assert nodes["target-1-prediction"]["transformation"] == "Sigmoid"
    assert nodes["resource-0"]["resourceClass"] == "PositiveScalarPerObservation"
    assert nodes["direct-component-0"]["componentIdentity"] == "direct.location"
    assert nodes["auxiliary-component-2"]["componentIdentity"] == "aux.risk-adjusted"
    assert nodes["total-loss"]["aggregation"] == "WeightedSum"

    input_ports = {
        node_id: {port["id"] for port in node["inputPorts"]}
        for node_id, node in nodes.items()
    }
    output_ports = {
        node_id: {port["id"] for port in node["outputPorts"]}
        for node_id, node in nodes.items()
    }
    for edge in topology["edges"]:
        assert edge["from"]["portId"] in output_ports[edge["from"]["nodeId"]]
        assert edge["to"]["portId"] in input_ports[edge["to"]["nodeId"]]

    assert {
        "from": {"nodeId": "resource-0", "portId": "value"},
        "to": {
            "nodeId": "auxiliary-component-2",
            "portId": "uncertaintyScale",
        },
        "logicalShape": [{"axis": "batch", "size": None}],
        "gradientFlow": "stopped",
    } in topology["edges"]


def test_topology_exposes_derived_positive_class_logit_correction():
    model = _weighted_binary_model()
    topology = ModelTopologyBuilder().build(model)

    validate_model_topology_document(
        {
            "requestId": "11111111-1111-4111-8111-111111111111",
            "modelRef": model.model_ref,
            **topology,
        },
        "detail-result",
    )

    nodes = {node["id"]: node for node in topology["nodes"]}
    correction = nodes["target-0-public-correction"]
    assert correction["positiveClassWeight"] == 28.0
    assert correction["derivedCorrection"] == "subtractLogPositiveClassWeight"
    assert {
        "from": {
            "nodeId": "target-0-public-correction",
            "portId": "corrected",
        },
        "to": {
            "nodeId": "target-0-prediction",
            "portId": "raw",
        },
        "logicalShape": [{"axis": "batch", "size": None}],
        "gradientFlow": "propagates",
    } in topology["edges"]


def test_topology_response_is_valid_for_its_query_and_flight_action_result():
    model = _model()
    topology = ModelTopologyBuilder().build(model)
    document = {
        "requestId": "11111111-1111-4111-8111-111111111111",
        "modelRef": model.model_ref,
        **topology,
    }

    validate_model_topology_document(document, "detail-result")
    validate_request_document(document, "action-result")


def test_flight_v18_dispatches_the_model_topology_action():
    model = _model()
    entry = CatalogModelRecord(
        model=model,
        catalog_ordinal=1,
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
    )
    store = SimpleNamespace(
        get_model=lambda owner, model_ref: (
            entry
            if (owner, model_ref) == ("owner-a", model.model_ref)
            else None
        )
    )
    topology = GetModelTopology(
        store,
        metadata_verifier=SimpleNamespace(verify=lambda _model: None),
        builder=ModelTopologyBuilder(),
    )
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
        get_model_topology=topology,
        service_status=SimpleNamespace(),
        availability=SimpleNamespace(),
    )
    document = {
        "requestId": "11111111-1111-4111-8111-111111111111",
        "modelRef": model.model_ref,
    }

    assert DETAIL_ACTION in ACTIONS
    request = validate_action_request(DETAIL_ACTION, document)
    response = json.loads(
        coordinator.dispatch(DETAIL_ACTION, "owner-a", request, document)
    )

    assert response["modelRef"] == model.model_ref
    assert response["topologyRevision"] == 2


def test_topology_query_is_owner_scoped_and_does_not_read_the_checkpoint():
    model = _model()
    entry = CatalogModelRecord(
        model=model,
        catalog_ordinal=1,
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
    )
    verified: list[str] = []
    store = SimpleNamespace(
        get_model=lambda owner, model_ref: (
            entry
            if (owner, model_ref) == ("owner-a", model.model_ref)
            else None
        )
    )
    query = GetModelTopology(
        store,
        metadata_verifier=SimpleNamespace(
            verify=lambda value: verified.append(value.model_ref)
        ),
        builder=ModelTopologyBuilder(),
    )

    result = query.execute(
        GetModelTopologyQuery(
            owner_subject="owner-a",
            request_id="11111111-1111-4111-8111-111111111111",
            model_ref=model.model_ref,
        )
    )

    assert result.model_ref == model.model_ref
    assert result.model_definition_sha256 == model.semantic_digests[
        "modelDefinitionSha256"
    ]
    assert verified == [model.model_ref]
    response = json.loads(model_topology_response(present_model_topology(result)))
    assert response["modelRef"] == model.model_ref

    with pytest.raises(CatalogModelNotFound):
        query.execute(
            GetModelTopologyQuery(
                owner_subject="owner-b",
                request_id="11111111-1111-4111-8111-111111111111",
                model_ref=model.model_ref,
            )
        )


def test_topology_query_hides_a_generation_deleted_while_it_is_built():
    model = _model()
    entry = CatalogModelRecord(
        model=model,
        catalog_ordinal=1,
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
    )
    calls = 0

    def get_model(owner: str, model_ref: str):
        nonlocal calls
        calls += 1
        if calls == 1 and (owner, model_ref) == ("owner-a", model.model_ref):
            return entry
        return None

    query = GetModelTopology(
        SimpleNamespace(get_model=get_model),
        metadata_verifier=SimpleNamespace(verify=lambda _model: None),
        builder=ModelTopologyBuilder(),
    )

    with pytest.raises(CatalogModelNotFound):
        query.execute(
            GetModelTopologyQuery(
                owner_subject="owner-a",
                request_id="11111111-1111-4111-8111-111111111111",
                model_ref=model.model_ref,
            )
        )


def test_topology_rejects_a_model_that_exceeds_its_node_limit():
    model = _model()
    oversized_config = ModelConfig(
        seq_len=3,
        feature_dim=12,
        hidden=24,
        layers=1_020,
        dropout=0.1,
        nhead=4,
        context_mode="relaxed",
    )
    contract = ModelContract.from_document(model.model_contract)
    oversized = replace(
        model,
        metadata={"modelConfig": oversized_config.to_manifest()},
        semantic_digests=resolved_semantic_digests(
            contract,
            "a" * 64,
            oversized_config,
        ),
    )

    with pytest.raises(OverflowError, match="node limit"):
        ModelTopologyBuilder().build(oversized)
