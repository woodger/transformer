from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from app.contracts.semantic.v1 import ModelContract
from app.contracts.training_telemetry.v1 import (
    TrainingTelemetryContractError,
    training_telemetry_capabilities,
    validate_training_telemetry_document,
)
from app.project import PROJECT_ROOT

TELEMETRY_ROOT = PROJECT_ROOT / "app" / "contracts" / "training_telemetry" / "v1"
SEMANTIC_ROOT = PROJECT_ROOT / "app" / "contracts" / "semantic" / "v1"
SCHEMA_ROOTS = (TELEMETRY_ROOT / "schemas", SEMANTIC_ROOT / "schemas")
FIXTURE_ROOT = TELEMETRY_ROOT / "fixtures"

ERROR_REASONS = {
    "INVALID_TELEMETRY_QUERY",
    "INVALID_TELEMETRY_CURSOR",
    "TELEMETRY_CURSOR_EXPIRED",
    "TELEMETRY_CURSOR_INVALIDATED",
    "TELEMETRY_QUERY_REVISION_UNAVAILABLE",
    "MODEL_NOT_FOUND",
    "STORED_MODEL_METADATA_INVALID",
    "TRAINING_TELEMETRY_INTEGRITY_FAILED",
    "TRAINING_TELEMETRY_BACKEND_UNAVAILABLE",
    "TELEMETRY_RESPONSE_BUDGET_EXCEEDED",
    "TELEMETRY_SNAPSHOT_CAPACITY_EXHAUSTED",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture(name: str) -> dict:
    return _read(FIXTURE_ROOT / name)


def _schema_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(path for root in SCHEMA_ROOTS for path in root.glob("*.schema.json"))
    )


@pytest.fixture(scope="module")
def schema_registry() -> Registry:
    resources = []
    for path in _schema_paths():
        schema = _read(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(schema_name: str, registry: Registry) -> Draft202012Validator:
    return Draft202012Validator(
        _read(TELEMETRY_ROOT / "schemas" / schema_name),
        registry=registry,
        format_checker=FormatChecker(),
    )


def _validate(document: object, schema_name: str, registry: Registry) -> None:
    _validator(schema_name, registry).validate(document)


def _fixture_schema(name: str) -> str:
    if name == "capabilities.json":
        return "capabilities.schema.json"
    if name.startswith("report.request."):
        return "report-request.schema.json"
    if name.startswith("report.result."):
        return "report-result.schema.json"
    if name.startswith("gradient-interactions.request."):
        return "gradient-interactions-request.schema.json"
    if name.startswith("gradient-interactions.result."):
        return "gradient-interactions-result.schema.json"
    if name.startswith("error."):
        return "error-detail.schema.json"
    raise AssertionError(f"fixture has no schema mapping: {name}")


def _assert_finite_numbers(value: object) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(value)
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite_numbers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item)


def _assert_health(health: dict) -> None:
    completed = health["trainingBatchesCompleted"]
    assert completed == (
        health["optimizerUpdatesApplied"] + health["optimizerUpdatesSkipped"]
    )
    assert completed == (
        health["finiteGradientBatches"] + health["nonFiniteGradientBatches"]
    )
    assert health["ampOverflowBatches"] <= health["optimizerUpdatesSkipped"]
    assert health["ampOverflowBatches"] <= health["nonFiniteGradientBatches"]


def _without_roles(anchor: dict) -> dict:
    result = dict(anchor)
    result.pop("roles")
    return result


def _assert_full_report(report: dict, items: list[dict]) -> None:
    coverage = report["coverage"]
    completed = coverage["completedEpochs"]
    assert coverage == {
        "completedEpochs": completed,
        "firstEpoch": 1,
        "lastEpoch": completed,
    }
    assert [item["epoch"] for item in items] == list(range(1, completed + 1))
    assert all(
        later["globalStep"] > earlier["globalStep"]
        for earlier, later in zip(items, items[1:], strict=False)
    )

    selection = report["selection"]
    if selection["enabled"]:
        assert selection["bestEpoch"] == selection["publishedEpoch"]
        assert selection["publishedSource"] == "best_direct_selection_score"
        assert all(item["selectionScore"] is not None for item in items)
    else:
        assert selection["bestEpoch"] is None
        assert selection["publishedEpoch"] == completed
        assert selection["publishedSource"] == "last_epoch"
        assert all(item["selectionScore"] is None for item in items)

    expected_roles: dict[int, list[str]] = {1: ["first"]}
    if selection["bestEpoch"] is not None:
        expected_roles.setdefault(selection["bestEpoch"], []).append("best")
    expected_roles.setdefault(selection["publishedEpoch"], []).append("published")
    anchors = report["anchors"]
    assert [anchor["epoch"] for anchor in anchors] == sorted(expected_roles)
    item_by_epoch = {item["epoch"]: item for item in items}
    for anchor in anchors:
        assert anchor["roles"] == expected_roles[anchor["epoch"]]
        assert _without_roles(anchor) == item_by_epoch[anchor["epoch"]]

    total_health = {name: 0 for name in report["healthTotals"]}
    target_layout = None
    direct_layout = None
    auxiliary_layout = None
    for item in items:
        _assert_health(item["health"])
        for name, value in item["health"].items():
            total_health[name] += value
        current_targets = [
            (metric["targetIndex"], metric["targetIdentity"])
            for metric in item["targetMetrics"]
        ]
        current_direct = [
            (loss["targetIndex"], loss["targetIdentity"])
            for loss in item["directLosses"]
        ]
        current_auxiliary = [
            loss["componentIdentity"] for loss in item["auxiliaryLosses"]
        ]
        assert [index for index, _identity in current_targets] == list(
            range(len(current_targets))
        )
        assert current_direct == current_targets
        assert current_auxiliary == sorted(current_auxiliary)
        target_layout = target_layout or current_targets
        direct_layout = direct_layout or current_direct
        auxiliary_layout = auxiliary_layout or current_auxiliary
        assert current_targets == target_layout
        assert current_direct == direct_layout
        assert current_auxiliary == auxiliary_layout
    assert total_health == report["healthTotals"]
    _assert_health(report["healthTotals"])

    collected = [
        item["epoch"] for item in items if item["gradientInteractionsCollected"]
    ]
    gradient = report["gradientInteractions"]
    if not collected:
        assert gradient["state"] in {
            "notConfigured",
            "configuredWithoutObservations",
        }
    else:
        published = selection["publishedEpoch"]
        previous = [epoch for epoch in collected if epoch < published]
        following = [epoch for epoch in collected if epoch > published]
        expected_default = (
            published
            if published in collected
            else max(previous)
            if previous
            else min(following)
        )
        assert gradient == {
            "state": "available",
            "collectedEpochCount": len(collected),
            "defaultEpoch": expected_default,
            "publishedEpochCollected": published in collected,
        }
    _assert_finite_numbers(report)


def test_schemas_are_valid_draft_2020_12_documents() -> None:
    for path in _schema_paths():
        Draft202012Validator.check_schema(_read(path))


def test_all_golden_documents_match_their_schemas(
    schema_registry: Registry,
) -> None:
    paths = sorted(
        path
        for path in FIXTURE_ROOT.glob("*.json")
        if path.name != "manifest.json"
    )
    for path in paths:
        _validate(_read(path), _fixture_schema(path.name), schema_registry)


def test_capabilities_fix_query_semantics_without_activating_flight() -> None:
    capabilities = _fixture("capabilities.json")
    assert capabilities == training_telemetry_capabilities()
    assert capabilities["epochObservation"] == {
        "kind": "PreOptimizerUpdateEpochPass",
        "checkpointReevaluation": False,
    }
    assert capabilities["outcomePrecedence"] == [
        "ModelLookup",
        "Pending",
        "Unavailable",
        "IntegrityFailure",
        "Available",
    ]
    assert capabilities["features"]["partialReports"] is False
    assert capabilities["features"]["checkpointVerification"] is False


def test_selection_disabled_report_has_bounded_epoch_pagination() -> None:
    initial = _fixture("report.request.initial.json")
    continuation = _fixture("report.request.continuation.json")
    first = _fixture("report.result.available.selection-disabled.first.json")
    terminal = _fixture("report.result.available.selection-disabled.terminal.json")

    assert initial["cursor"] is None
    assert first["requestId"] == initial["requestId"]
    assert first["modelRef"] == initial["modelRef"]
    assert len(first["epochPage"]["items"]) == initial["pageSize"]
    assert continuation["cursor"] == first["epochPage"]["nextCursor"]
    assert continuation["pageSize"] == initial["pageSize"]
    assert terminal["requestId"] == continuation["requestId"]
    assert terminal["epochPage"]["nextCursor"] is None
    assert terminal["epochPage"]["cursorExpiresAt"] is None

    immutable_fields = (
        "modelRef",
        "producingRunId",
        "semanticDigests",
        "coverage",
        "selection",
        "anchors",
        "healthTotals",
        "gradientInteractions",
    )
    assert all(first[name] == terminal[name] for name in immutable_fields)
    items = first["epochPage"]["items"] + terminal["epochPage"]["items"]
    _assert_full_report(first, items)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "report.result.available.selection-enabled.json",
        "report.result.available.single-epoch.json",
        "report.result.available.configured-without-observations.json",
        "report.result.available.default-previous.json",
        "report.result.available.default-following.json",
    ],
)
def test_terminal_available_fixtures_obey_full_report_invariants(
    fixture_name: str,
) -> None:
    report = _fixture(fixture_name)
    assert report["epochPage"]["nextCursor"] is None
    assert report["epochPage"]["cursorExpiresAt"] is None
    _assert_full_report(report, report["epochPage"]["items"])


def test_multi_target_report_resolves_to_consumer_neutral_model_contract() -> None:
    report = _fixture("report.result.available.selection-enabled.json")
    semantic_fixture = _read(
        SEMANTIC_ROOT / "fixtures" / "multi-target-shared-resource.json"
    )
    model_contract = ModelContract.from_document(semantic_fixture["modelContract"])
    assert report["semanticDigests"] == model_contract.digests("c" * 64)

    target_layout = list(enumerate(model_contract.target_identities))
    direct = [
        (item["identity"], item["operator"])
        for item in model_contract.direct_components
    ]
    auxiliary = [
        (item["identity"], item["operator"])
        for item in model_contract.auxiliary_components
    ]
    for epoch in report["epochPage"]["items"]:
        assert [
            (item["targetIndex"], item["targetIdentity"])
            for item in epoch["targetMetrics"]
        ] == target_layout
        assert [
            (item["componentIdentity"], item["operator"])
            for item in epoch["directLosses"]
        ] == direct
        assert [
            (item["componentIdentity"], item["operator"])
            for item in epoch["auxiliaryLosses"]
        ] == auxiliary


def test_authoritative_aggregates_do_not_require_cross_field_equality() -> None:
    report = _fixture("report.result.available.selection-enabled.json")
    semantic_fixture = _read(
        SEMANTIC_ROOT / "fixtures" / "multi-target-shared-resource.json"
    )
    model_contract = ModelContract.from_document(semantic_fixture["modelContract"])
    weights = {
        item["identity"]: item["weight"]
        for item in (*model_contract.direct_components, *model_contract.auxiliary_components)
    }
    first_epoch = report["epochPage"]["items"][0]
    reconstructed_total = math.fsum(
        weights[item["componentIdentity"]] * item["value"]
        for family in (first_epoch["directLosses"], first_epoch["auxiliaryLosses"])
        for item in family
    )
    assert first_epoch["totalLoss"] != reconstructed_total


def test_default_epoch_covers_published_previous_and_following_branches() -> None:
    published = _fixture("report.result.available.selection-enabled.json")
    previous = _fixture("report.result.available.default-previous.json")
    following = _fixture("report.result.available.default-following.json")
    assert published["gradientInteractions"] == {
        "state": "available",
        "collectedEpochCount": 3,
        "defaultEpoch": 2,
        "publishedEpochCollected": True,
    }
    assert previous["gradientInteractions"]["defaultEpoch"] == 1
    assert following["gradientInteractions"]["defaultEpoch"] == 2


def test_gradient_pairs_are_oriented_unique_ordered_and_bounded() -> None:
    initial = _fixture("gradient-interactions.request.initial.json")
    continuation = _fixture("gradient-interactions.request.continuation.json")
    first = _fixture("gradient-interactions.result.available.first.json")
    terminal = _fixture("gradient-interactions.result.available.terminal.json")
    components = [item["componentIdentity"] for item in first["components"]]

    assert first["components"] == terminal["components"]
    assert len(first["pairPage"]["items"]) == initial["pageSize"]
    assert first["pairPage"]["nextCursor"] == continuation["cursor"]
    assert initial["pageSize"] == continuation["pageSize"]
    assert terminal["pairPage"]["nextCursor"] is None
    assert terminal["pairPage"]["cursorExpiresAt"] is None

    pairs = first["pairPage"]["items"] + terminal["pairPage"]["items"]
    pair_identities = [
        (item["leftComponentIdentity"], item["rightComponentIdentity"])
        for item in pairs
    ]
    execution_position = {identity: index for index, identity in enumerate(components)}
    assert pair_identities == sorted(pair_identities)
    assert len(pair_identities) == len(set(pair_identities))
    assert set(pair_identities) <= set(combinations(components, 2))
    assert all(
        execution_position[left] < execution_position[right]
        for left, right in pair_identities
    )
    _assert_finite_numbers(first)
    _assert_finite_numbers(terminal)


def test_zero_norm_produces_a_sparse_gradient_pair_result() -> None:
    result = _fixture("gradient-interactions.result.zero-norm-sparse.json")
    components = [item["componentIdentity"] for item in result["components"]]
    pair_identities = [
        (item["leftComponentIdentity"], item["rightComponentIdentity"])
        for item in result["pairPage"]["items"]
    ]
    pairs = set(pair_identities)
    zero_component = result["components"][-1]

    assert pair_identities == sorted(pair_identities)
    assert len(pair_identities) == len(pairs)
    assert zero_component == {
        "componentIdentity": "aux.risk-adjusted",
        "meanNorm": 0.0,
    }
    assert pairs == {
        pair
        for pair in combinations(components, 2)
        if zero_component["componentIdentity"] not in pair
    }


def test_absent_gradient_observations_are_normal_results() -> None:
    not_configured = _fixture("gradient-interactions.result.not-configured.json")
    not_collected = _fixture("gradient-interactions.result.not-collected.json")
    assert not_configured["state"] == "notCollected"
    assert not_configured["reason"] == "DIAGNOSTICS_NOT_CONFIGURED"
    assert not_collected["state"] == "notCollected"
    assert not_collected["reason"] == "NO_OBSERVATIONS_FOR_EPOCH"


def test_report_outcomes_distinguish_progress_final_absence_and_corruption() -> None:
    assert _fixture("report.result.pending.json")["reason"] == (
        "MATERIALIZATION_PENDING"
    )
    assert _fixture("report.result.unavailable.no-complete-report.json")[
        "reason"
    ] == "NO_COMPLETE_REPORT"
    assert _fixture("error.telemetry-integrity.missing-epoch.json")["reason"] == (
        "TRAINING_TELEMETRY_INTEGRITY_FAILED"
    )


def test_unknown_and_foreign_models_are_security_equivalent() -> None:
    assert _fixture("error.model-not-found.unknown.json") == _fixture(
        "error.model-not-found.foreign.json"
    )
    deleted = _fixture("error.model-not-found.deleted-during-cursor.json")
    assert deleted["reason"] == "MODEL_NOT_FOUND"


def test_every_structured_error_outcome_has_a_valid_fixture(
    schema_registry: Registry,
) -> None:
    reasons = set()
    for path in sorted(FIXTURE_ROOT.glob("error.*.json")):
        document = _read(path)
        _validate(document, "error-detail.schema.json", schema_registry)
        reasons.add(document["reason"])
    assert reasons == ERROR_REASONS


def test_codec_accepts_contract_documents_and_reports_exact_paths() -> None:
    capabilities = _fixture("capabilities.json")
    assert validate_training_telemetry_document(
        capabilities, "capabilities"
    ) == capabilities
    invalid = deepcopy(_fixture("report.request.initial.json"))
    invalid["pageSize"] = 101
    with pytest.raises(TrainingTelemetryContractError, match=r"^/pageSize:"):
        validate_training_telemetry_document(invalid, "report-request")


def test_revision_one_rejects_unadvertised_query_shapes(
    schema_registry: Registry,
) -> None:
    oversized_epoch_page = deepcopy(_fixture("report.request.initial.json"))
    oversized_epoch_page["pageSize"] = 101
    with pytest.raises(ValidationError):
        _validate(
            oversized_epoch_page,
            "report-request.schema.json",
            schema_registry,
        )

    oversized_pair_page = deepcopy(
        _fixture("gradient-interactions.request.initial.json")
    )
    oversized_pair_page["pageSize"] = 1001
    with pytest.raises(ValidationError):
        _validate(
            oversized_pair_page,
            "gradient-interactions-request.schema.json",
            schema_registry,
        )

    partial_cursor_state = deepcopy(
        _fixture("report.result.available.selection-disabled.first.json")
    )
    partial_cursor_state["epochPage"]["cursorExpiresAt"] = None
    with pytest.raises(ValidationError):
        _validate(
            partial_cursor_state,
            "report-result.schema.json",
            schema_registry,
        )

    provider_storage = deepcopy(_fixture("report.result.pending.json"))
    provider_storage["indexName"] = "metrics-runs-v5"
    with pytest.raises(ValidationError):
        _validate(provider_storage, "report-result.schema.json", schema_registry)


def test_offline_fixture_manifest_has_exact_sorted_file_digests(
    schema_registry: Registry,
) -> None:
    manifest = _fixture("manifest.json")
    _validate(manifest, "fixture-manifest.schema.json", schema_registry)

    paths = [entry["path"] for entry in manifest["files"]]
    actual_paths = sorted(
        path.name for path in FIXTURE_ROOT.glob("*.json") if path.name != "manifest.json"
    )
    assert paths == sorted(paths)
    assert paths == actual_paths
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((FIXTURE_ROOT / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )

    cases = manifest["cases"]
    identities = [case["identity"] for case in cases]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities))
    referenced = {
        document for case in cases for document in case["documents"]
    }
    assert referenced == set(paths)

    manifest_digest, manifest_name = (FIXTURE_ROOT / "manifest.sha256").read_text(
        encoding="ascii"
    ).split()
    assert manifest_name == "manifest.json"
    assert manifest_digest == hashlib.sha256(
        (FIXTURE_ROOT / manifest_name).read_bytes()
    ).hexdigest()
