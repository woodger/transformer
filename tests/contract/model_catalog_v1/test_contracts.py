from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from app.contracts.semantic.v1 import ModelContract
from app.project import PROJECT_ROOT

CATALOG_ROOT = PROJECT_ROOT / "app" / "contracts" / "model_catalog" / "v1"
SEMANTIC_ROOT = PROJECT_ROOT / "app" / "contracts" / "semantic" / "v1"
CHECKPOINT_ROOT = PROJECT_ROOT / "app" / "contracts" / "checkpoint" / "v6"
SCHEMA_ROOTS = (
    CATALOG_ROOT / "schemas",
    SEMANTIC_ROOT / "schemas",
    CHECKPOINT_ROOT / "schemas",
)
FIXTURE_ROOT = CATALOG_ROOT / "fixtures"

FIXTURE_SCHEMAS = {
    "capabilities.json": "capabilities.schema.json",
    "detail.request.json": "detail-request.schema.json",
    "detail.request.published-model.json": "detail-request.schema.json",
    "detail.result.published-model.json": "detail-result.schema.json",
    "detail.result.random.json": "detail-result.schema.json",
    "list.request.continuation.json": "list-request.schema.json",
    "list.request.initial.json": "list-request.schema.json",
    "list.result.empty.json": "list-result.schema.json",
    "list.result.first.json": "list-result.schema.json",
    "list.result.terminal.json": "list-result.schema.json",
}
ERROR_REASONS = {
    "CATALOG_CURSOR_EXPIRED",
    "CATALOG_QUERY_REVISION_UNAVAILABLE",
    "CATALOG_RESPONSE_BUDGET_EXCEEDED",
    "CHECKPOINT_VERIFICATION_BUDGET_EXCEEDED",
    "INVALID_CATALOG_CURSOR",
    "INVALID_CATALOG_QUERY",
    "MODEL_CHECKPOINT_DIGEST_MISMATCH",
    "MODEL_CHECKPOINT_SIZE_MISMATCH",
    "MODEL_CHECKPOINT_UNAVAILABLE",
    "MODEL_NOT_FOUND",
    "MODEL_REGISTRY_UNAVAILABLE",
    "STORED_MODEL_METADATA_INVALID",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        _read(CATALOG_ROOT / "schemas" / schema_name),
        registry=registry,
        format_checker=FormatChecker(),
    )


def _validate(document: object, schema_name: str, registry: Registry) -> None:
    _validator(schema_name, registry).validate(document)


def _fixture(name: str) -> dict:
    return _read(FIXTURE_ROOT / name)


def test_schemas_are_valid_draft_2020_12_documents() -> None:
    for path in _schema_paths():
        Draft202012Validator.check_schema(_read(path))


@pytest.mark.parametrize(
    ("fixture_name", "schema_name"),
    sorted(FIXTURE_SCHEMAS.items()),
)
def test_request_result_and_capability_fixtures_match_schemas(
    fixture_name: str,
    schema_name: str,
    schema_registry: Registry,
) -> None:
    _validate(_fixture(fixture_name), schema_name, schema_registry)


def test_every_structured_error_outcome_has_a_valid_fixture(
    schema_registry: Registry,
) -> None:
    fixtures = sorted(FIXTURE_ROOT.glob("error.*.json"))
    reasons = set()
    for path in fixtures:
        document = _read(path)
        _validate(document, "error-detail.schema.json", schema_registry)
        reasons.add(document["reason"])
    assert reasons == ERROR_REASONS


def test_list_fixtures_define_bounded_live_high_water_traversal() -> None:
    capabilities = _fixture("capabilities.json")
    initial = _fixture("list.request.initial.json")
    continuation = _fixture("list.request.continuation.json")
    first = _fixture("list.result.first.json")
    terminal = _fixture("list.result.terminal.json")
    empty = _fixture("list.result.empty.json")

    assert capabilities["consistency"] == {
        "kind": "LiveHighWater",
        "ordering": [
            {"field": "createdAt", "direction": "DESC"},
            {"field": "modelRef", "direction": "ASC"},
        ],
        "concurrentPublication": "ExcludedAfterHighWater",
        "concurrentDeletion": "MayOmit",
        "cursorExpiration": "RestartTraversal",
    }
    assert initial["cursor"] is None
    assert first["requestId"] == initial["requestId"]
    assert len(first["models"]) == initial["pageSize"]
    assert continuation["cursor"] == first["nextCursor"]
    assert continuation["pageSize"] == initial["pageSize"]
    assert terminal["requestId"] == continuation["requestId"]
    assert terminal["nextCursor"] is None
    assert terminal["cursorExpiresAt"] is None
    assert empty["models"] == []
    assert empty["nextCursor"] is None
    assert empty["cursorExpiresAt"] is None

    expires_at = datetime.fromisoformat(first["cursorExpiresAt"])
    created_at = datetime.fromisoformat(first["models"][0]["createdAt"])
    assert int((expires_at - created_at).total_seconds()) == 900

    ordered_models = first["models"] + terminal["models"]
    assert [model["modelRef"] for model in ordered_models] == [
        "mdl_11111111111111111111111111111111",
        "mdl_22222222222222222222222222222222",
        "mdl_33333333333333333333333333333333",
    ]


@pytest.mark.parametrize(
    ("request_name", "result_name"),
    [
        ("detail.request.json", "detail.result.random.json"),
        (
            "detail.request.published-model.json",
            "detail.result.published-model.json",
        ),
    ],
)
def test_detail_is_an_exact_verified_projection_of_model_metadata(
    request_name: str,
    result_name: str,
) -> None:
    request = _fixture(request_name)
    result = _fixture(result_name)
    detail = result["model"]
    summary = detail["summary"]
    model_contract = ModelContract.from_document(detail["modelContract"])
    data_digest = detail["dataContract"]["dataContractSha256"]
    digests = model_contract.digests(data_digest)

    first_page = _fixture("list.result.first.json")
    list_summary = next(
        item for item in first_page["models"] if item["modelRef"] == request["modelRef"]
    )
    assert result["requestId"] == request["requestId"]
    assert summary == list_summary
    assert summary["modelConfig"] == model_contract.model_config
    assert summary["targetIdentities"] == list(model_contract.target_identities)
    assert summary["semanticDigests"] == digests
    assert detail["selection"]["modelContractSha256"] == digests[
        "modelContractSha256"
    ]
    assert detail["dataContract"]["seqLen"] == summary["modelConfig"]["seqLen"]
    assert detail["dataContract"]["featureDim"] == summary["modelConfig"][
        "featureDim"
    ]

    initialization = detail["initialization"]
    expected_summary = {"kind": initialization["kind"]}
    if initialization["kind"] == "publishedModel":
        expected_summary["parentModelRef"] = initialization["parentModelRef"]
        assert initialization["dataContractSha256"] == data_digest
        assert initialization["parentDataContractSha256"] == data_digest
        for name in (
            "targetContractSha256",
            "objectiveSha256",
            "modelContractSha256",
        ):
            assert initialization[name] == digests[name]
            assert initialization[f"parent{name[0].upper()}{name[1:]}"] == digests[name]
    assert summary["initialization"] == expected_summary

    limits = _fixture("capabilities.json")["limits"]
    assert summary["checkpoint"]["bytes"] <= limits[
        "maxCheckpointVerificationBytes"
    ]
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= limits[
        "maxResponseBytes"
    ]


def test_revision_one_rejects_unadvertised_query_shapes(
    schema_registry: Registry,
) -> None:
    oversized_page = deepcopy(_fixture("list.request.initial.json"))
    oversized_page["pageSize"] = 101
    with pytest.raises(ValidationError):
        _validate(oversized_page, "list-request.schema.json", schema_registry)

    partial_cursor_state = deepcopy(_fixture("list.result.first.json"))
    partial_cursor_state["cursorExpiresAt"] = None
    with pytest.raises(ValidationError):
        _validate(partial_cursor_state, "list-result.schema.json", schema_registry)

    batch_detail = deepcopy(_fixture("detail.request.json"))
    batch_detail["modelRefs"] = [batch_detail.pop("modelRef")]
    with pytest.raises(ValidationError):
        _validate(batch_detail, "detail-request.schema.json", schema_registry)


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

    manifest_digest, manifest_name = (FIXTURE_ROOT / "manifest.sha256").read_text(
        encoding="ascii"
    ).split()
    assert manifest_name == "manifest.json"
    assert manifest_digest == hashlib.sha256(
        (FIXTURE_ROOT / manifest_name).read_bytes()
    ).hexdigest()
