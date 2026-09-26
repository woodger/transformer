import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app.contracts.checkpoint.v11.codec import validate_checkpoint_document
from app.contracts.flight.v21.codec import validate_request_document
from app.contracts.flight.v21.job_config import job_config_sha256
from app.contracts.model_catalog.v6.codec import validate_catalog_document
from app.contracts.target_head_diagnostics.v4.codec import (
    validate_target_head_diagnostics_document,
)
from app.contracts.worker.v19.codec import validate_document
from app.contracts.worker.v19.diagnostics import DiagnosticsConfig
from app.project import PROJECT_ROOT


def test_staged_encoder_layer_diagnostics_contract_fixtures_are_valid():
    _validate_target_head_diagnostics_fixtures()
    _validate_model_catalog_fixtures()
    _validate_flight_fixtures()


def test_staged_internal_contract_schemas_are_loadable():
    valid_diagnostics = DiagnosticsConfig(
        target_head="fullCommittedArtifact",
        encoder_layer_diagnostics="directComponentPerBatch",
    )
    assert valid_diagnostics.to_document() == {
        "schemaVersion": 3,
        "gradientInteractions": None,
        "targetHead": "fullCommittedArtifact",
        "encoderLayerDiagnostics": "directComponentPerBatch",
    }

    with pytest.raises(ValueError, match="require full target head"):
        DiagnosticsConfig(encoder_layer_diagnostics="directComponentPerBatch")

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


def _validate_target_head_diagnostics_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/target_head_diagnostics/v4/fixtures"
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


def _validate_model_catalog_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/model_catalog/v6/fixtures"
    _validate_manifest(
        root,
        lambda value: validate_catalog_document(value, "fixture-manifest"),
    )
    for path in _fixture_paths(root):
        schema = "detail" if path.name.startswith("detail") else "list"
        direction = "request" if ".request." in path.name else "result"
        document = _json(path)
        validate_catalog_document(document, f"{schema}-{direction}")
        if direction == "result":
            validate_request_document(document, "action-result")


def _validate_flight_fixtures() -> None:
    root = PROJECT_ROOT / "app/contracts/flight/v21/fixtures"
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
        elif path.name == "capabilities.result.json":
            schema_name = "capabilities-result"
        elif path.name.startswith("error."):
            schema_name = "error-detail"
        else:
            schema_name = "requested-initialization"
        validate_request_document(document, schema_name)
        if schema_name == "job-config-fixture":
            assert job_config_sha256(document["jobConfig"]) == document[
                "expectedJobConfigSha256"
            ]


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
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]

    expected_digest = (root / "manifest.sha256").read_text(
        encoding="utf-8",
    ).split()[0]
    assert hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == (
        expected_digest
    )


def _fixture_paths(root: Path) -> list[Path]:
    manifest = _json(root / "manifest.json")
    return [root / item["path"] for item in manifest["files"]]


def _json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    assert isinstance(document, dict)
    return document
