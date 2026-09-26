import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app.contracts.checkpoint.v12.codec import validate_checkpoint_document
from app.contracts.flight.v22.codec import validate_request_document
from app.contracts.flight.v22.job_config import job_config_sha256
from app.contracts.model_catalog.v7.codec import validate_catalog_document
from app.contracts.model_topology.v3.codec import (
    validate_model_topology_document,
)
from app.contracts.semantic.v5 import ModelContract, SemanticContractError
from app.contracts.semantic.v5.schema import validate_schema
from app.contracts.target_head_diagnostics.v5.codec import (
    validate_target_head_diagnostics_document,
)
from app.contracts.worker.v20.codec import validate_document
from app.contracts.worker.v20.model_config import ModelConfig
from app.contracts.worker.v20.model_definition import resolved_semantic_digests
from app.project import PROJECT_ROOT


def test_staged_encoder_normalization_order_contract_fixtures_are_valid():
    _validate_semantic_fixtures()
    _validate_flight_fixtures()
    _validate_catalog_fixtures()
    _validate_topology_fixtures()
    _validate_target_head_diagnostics_fixtures()


def test_staged_model_definitions_bind_normalization_order():
    tuning = {
        "hiddenWidth": 16,
        "encoderLayerCount": 3,
        "attentionHeadCount": 4,
        "dropoutProbability": 0,
        "missingValuePolicy": "strict",
    }
    post_config = ModelConfig.from_tuning(
        {**tuning, "encoderNormalizationOrder": "postNorm"},
        seq_len=2,
        feature_dim=8,
    )
    pre_config = ModelConfig.from_tuning(
        {**tuning, "encoderNormalizationOrder": "preNorm"},
        seq_len=2,
        feature_dim=8,
    )
    contract = ModelContract.from_document(_binary_model_contract(tuning))

    post_definition = resolved_semantic_digests(
        contract,
        "a" * 64,
        post_config,
    )
    pre_definition = resolved_semantic_digests(
        contract,
        "a" * 64,
        pre_config,
    )

    assert post_config.to_manifest()["encoderNormalizationOrder"] == "postNorm"
    assert pre_config.to_manifest()["encoderNormalizationOrder"] == "preNorm"
    assert (
        post_definition["modelDefinitionSha256"]
        != pre_definition["modelDefinitionSha256"]
    )

    with pytest.raises(ValueError, match="unsupported or missing"):
        ModelConfig.from_tuning(tuning, seq_len=2, feature_dim=8)

    invalid_contract = _binary_model_contract(tuning)
    del invalid_contract["modelTuning"]["encoderNormalizationOrder"]
    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(invalid_contract)
    assert error.value.reason == "INVALID_MODEL_CONTRACT"
    assert error.value.path == "/modelTuning"


def test_staged_internal_schemas_require_complete_model_configuration():
    for schema_name in (
        "checkpoint-artifact",
        "checkpoint-metadata",
        "recovery-metadata",
        "resolved-initialization",
    ):
        with pytest.raises(ValueError):
            validate_checkpoint_document({}, schema_name)

    for schema_name in (
        "arrow-manifest",
        "capabilities",
        "command-manifest",
        "control-message",
        "event",
        "prediction-manifest",
        "recovery-descriptor",
        "result-manifest",
        "target-head-diagnostics-artifact",
        "training-metrics",
    ):
        with pytest.raises(ValueError):
            validate_document({}, schema_name)


def _validate_semantic_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/semantic/v5/fixtures"
    _validate_manifest(root, lambda value: validate_schema(value, "fixture-manifest"))

    for path in _fixture_paths(root):
        if path.suffix != ".json":
            continue
        document = _json(path)
        if path.name == "language-capabilities.json":
            validate_schema(document, "language-capabilities")
            continue
        validate_schema(document, "fixture")
        expected = document["expected"]
        model = ModelContract.from_document(document["modelContract"])
        digests = model.target_objective_digests()
        assert {
            "targetContractSha256": digests["targetContractSha256"],
            "objectiveSha256": digests["objectiveSha256"],
        } == {
            "targetContractSha256": expected["targetContractSha256"],
            "objectiveSha256": expected["objectiveSha256"],
        }


def _validate_flight_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/flight/v22/fixtures"
    _validate_manifest(
        root,
        lambda value: validate_request_document(value, "fixture-manifest"),
    )
    for path in _fixture_paths(root):
        document = _json(path)
        if path.parent.name == "indexed-feature-blocks":
            schema_name = "indexed-feature-blocks-fixture"
        elif path.name.startswith("job-config."):
            schema_name = "job-config-fixture"
        elif path.name.startswith("fit-create."):
            schema_name = "fit-create"
        elif path.name == "capabilities.result.json":
            schema_name = "capabilities-result"
        elif path.name.startswith("error."):
            schema_name = "error-detail"
        else:
            schema_name = "requested-initialization"
        validate_request_document(document, schema_name)
        if schema_name == "job-config-fixture":
            assert (
                job_config_sha256(document["jobConfig"])
                == document["expectedJobConfigSha256"]
            )


def _validate_catalog_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/model_catalog/v7/fixtures"
    _validate_manifest(
        root,
        lambda value: validate_catalog_document(value, "fixture-manifest"),
    )
    binary_definitions: dict[str, dict[str, object]] = {}
    for path in _fixture_paths(root):
        schema = "detail" if path.name.startswith("detail") else "list"
        direction = "request" if ".request." in path.name else "result"
        document = _json(path)
        validate_catalog_document(document, f"{schema}-{direction}")
        if direction == "result":
            validate_request_document(document, "action-result")
        if schema != "detail" or direction != "result":
            continue

        model = document["model"]
        contract = ModelContract.from_document(model["modelContract"])
        geometry = model["dataDefinition"]["tensorGeometry"]
        config = ModelConfig.from_tuning(
            model["modelContract"]["modelTuning"],
            seq_len=geometry["seqLen"],
            feature_dim=geometry["featureDim"],
        )
        digests = resolved_semantic_digests(
            contract,
            model["dataDefinition"]["dataContractSha256"],
            config,
        )
        assert model["semanticDigests"] == digests
        assert (
            model["summary"]["modelDefinitionSha256"]
            == digests["modelDefinitionSha256"]
        )

        if model["summary"]["label"].startswith("weighted-binary-model"):
            binary_definitions[config.normalization_order] = digests

    assert (
        binary_definitions["postNorm"]["targetContractSha256"]
        == (binary_definitions["preNorm"]["targetContractSha256"])
    )
    assert (
        binary_definitions["postNorm"]["objectiveSha256"]
        == (binary_definitions["preNorm"]["objectiveSha256"])
    )
    assert (
        binary_definitions["postNorm"]["modelDefinitionSha256"]
        != (binary_definitions["preNorm"]["modelDefinitionSha256"])
    )


def _validate_topology_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/model_topology/v3/fixtures"
    _validate_manifest(
        root,
        lambda value: validate_model_topology_document(value, "fixture-manifest"),
    )
    for path in _fixture_paths(root):
        document = _json(path)
        if path.name == "detail.request.json":
            schema_name = "detail-request"
        elif path.name.startswith("detail.result."):
            schema_name = "detail-result"
        else:
            schema_name = "error-detail"
        validate_model_topology_document(document, schema_name)
        if schema_name == "detail-result":
            validate_request_document(document, "action-result")


def _validate_target_head_diagnostics_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/target_head_diagnostics/v5/fixtures"
    _validate_manifest(
        root,
        lambda value: validate_target_head_diagnostics_document(
            value,
            "fixture-manifest",
        ),
    )
    for path in _fixture_paths(root):
        document = _json(path)
        schema_name = (
            "capabilities"
            if path.name == "capabilities.json"
            else "error-detail"
            if path.name.startswith("error.")
            else "report-request"
            if ".request." in path.name
            else "report-result"
        )
        validate_target_head_diagnostics_document(document, schema_name)
        if schema_name == "report-result":
            validate_request_document(document, "action-result")


def _validate_manifest(
    root: Path,
    validate: Callable[[dict[str, object]], object],
) -> None:
    manifest = _json(root / "manifest.json")
    validate(manifest)

    listed_paths = [item["path"] for item in manifest["files"]]
    assert listed_paths == sorted(listed_paths)
    assert set(listed_paths) == {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    for item in manifest["files"]:
        assert (
            hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )

    expected_digest = (
        (root / "manifest.sha256")
        .read_text(
            encoding="utf-8",
        )
        .split()[0]
    )
    assert hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == (
        expected_digest
    )


def _fixture_paths(root: Path) -> list[Path]:
    return [root / item["path"] for item in _json(root / "manifest.json")["files"]]


def _json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    assert isinstance(document, dict)
    return document


def _binary_model_contract(tuning: dict[str, object]) -> dict[str, object]:
    return {
        "targetContract": {
            "slots": [
                {
                    "identity": "OpaqueBinaryEvent",
                    "observedConstraint": {
                        "constraint": "ClosedInterval",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "lossInputTransformation": "Identity",
                    "publicPredictionTransformation": "Sigmoid",
                }
            ]
        },
        "objective": {
            "directComponents": [
                {
                    "identity": "direct.opaque-binary-event",
                    "operator": "BinaryCrossEntropyWithLogits",
                    "weight": 1,
                    "targetIdentity": "OpaqueBinaryEvent",
                }
            ]
        },
        "modelTuning": {**tuning, "encoderNormalizationOrder": "postNorm"},
    }
