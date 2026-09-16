import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from app.contracts.flight.v15.codec import validate_request_document
from app.contracts.flight.v15.job_config import job_config_sha256
from app.contracts.model_catalog.v3.codec import validate_catalog_document
from app.contracts.semantic.v3 import ModelContract
from app.contracts.semantic.v3.schema import validate_schema
from app.contracts.training_telemetry.v3.codec import (
    validate_training_telemetry_document,
)
from app.project import PROJECT_ROOT


def test_current_cross_project_fixtures_are_valid_and_intact():
    semantic_root = PROJECT_ROOT / "app/contracts/semantic/v3/fixtures"
    _validate_manifest(semantic_root, lambda value: validate_schema(
        value,
        "fixture-manifest",
    ))
    for path in _fixture_paths(semantic_root):
        if path.suffix != ".json":
            continue
        document = _json(path)
        if path.name == "language-capabilities.json":
            validate_schema(document, "language-capabilities")
            continue
        validate_schema(document, "fixture")
        model = ModelContract.from_document(document["modelContract"])
        digests = model.target_objective_digests()
        expected = document["expected"]
        assert {
            "targetContractSha256": digests["targetContractSha256"],
            "objectiveSha256": digests["objectiveSha256"],
        } == {
            "targetContractSha256": expected["targetContractSha256"],
            "objectiveSha256": expected["objectiveSha256"],
        }

    flight_root = PROJECT_ROOT / "app/contracts/flight/v15/fixtures"
    _validate_manifest(
        flight_root,
        lambda value: validate_request_document(value, "fixture-manifest"),
    )
    for path in _fixture_paths(flight_root):
        document = _json(path)
        if path.parts[-2] == "indexed-feature-blocks":
            validate_request_document(document, "indexed-feature-blocks-fixture")
        elif path.name.startswith("job-config."):
            validate_request_document(document, "job-config-fixture")
            assert job_config_sha256(document["jobConfig"]) == document[
                "expectedJobConfigSha256"
            ]
        elif path.name == "capabilities.result.json":
            validate_request_document(document, "capabilities-result")
        else:
            validate_request_document(document, "requested-initialization")

    catalog_root = PROJECT_ROOT / "app/contracts/model_catalog/v3/fixtures"
    _validate_manifest(
        catalog_root,
        lambda value: validate_catalog_document(value, "fixture-manifest"),
    )
    for path in _fixture_paths(catalog_root):
        schema = "detail" if path.name.startswith("detail") else "list"
        direction = "request" if ".request." in path.name else "result"
        document = _json(path)
        validate_catalog_document(document, f"{schema}-{direction}")
        if direction == "result":
            validate_request_document(document, "action-result")

    telemetry_root = PROJECT_ROOT / "app/contracts/training_telemetry/v3/fixtures"
    _validate_manifest(
        telemetry_root,
        lambda value: validate_training_telemetry_document(value, "fixture-manifest"),
    )
    for path in _fixture_paths(telemetry_root):
        document = _json(path)
        schema_name = _telemetry_schema_name(path)
        validate_training_telemetry_document(document, schema_name)
        if schema_name.endswith("result"):
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
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]

    expected_manifest_digest = (root / "manifest.sha256").read_text(
        encoding="utf-8",
    ).split()[0]
    assert hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == (
        expected_manifest_digest
    )


def _fixture_paths(root: Path) -> list[Path]:
    manifest = _json(root / "manifest.json")
    return [root / item["path"] for item in manifest["files"]]


def _telemetry_schema_name(path: Path) -> str:
    if path.name.startswith("report."):
        return "report-request" if ".request." in path.name else "report-result"
    return (
        "gradient-interactions-request"
        if ".request." in path.name
        else "gradient-interactions-result"
    )


def _json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    assert isinstance(document, dict)
    return document
