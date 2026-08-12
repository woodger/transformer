from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from flight_contract_schema import (
    check_contract_schema,
    read_contract_schema,
    validate_contract_document,
)
from jsonschema.exceptions import ValidationError

from app.contracts.worker.v3.config import (
    CheckpointSelectionConfig,
    TrainConfig,
)
from app.contracts.worker.v3.objective import (
    objective_config,
    objective_config_sha256,
)
from app.flight.arrow import schema_fingerprint
from app.flight.constants import (
    ACQUIRE_ACTION,
    ACTIONS,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CREATE_ACTION,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MODEL_DESCRIBE_ACTION,
    OUTPUTS_LIST_ACTION,
    STATUS_ACTION,
)
from app.flight.contract import (
    canonical_manifest_hash,
    parse_action_body,
    validate_action_request,
    validate_upload_metadata,
)

FIXTURE_ROOT = (
    Path(__file__).parents[1]
    / "app"
    / "contracts"
    / "flight"
    / "v4"
    / "fixtures"
)
JSON_ROOT = FIXTURE_ROOT / "json"
ARROW_ROOT = FIXTURE_ROOT / "arrow"
SCHEMA_ROOT = FIXTURE_ROOT.parent / "schemas"

REQUEST_FIXTURES = {
    "capabilities.request.json": "query.schema.json",
    "health.request.json": "query.schema.json",
    "create-fit.request.json": "create.schema.json",
    "create-predict.request.json": "create.schema.json",
    "acquire.request.json": "acquire.schema.json",
    "status.request.json": "status.schema.json",
    "inputs-list.request.json": "inputs-list.schema.json",
    "input-close.request.json": "input-close.schema.json",
    "outputs-list.request.json": "outputs-list.schema.json",
    "cancel.request.json": "cancel.schema.json",
    "model-describe.request.json": "model-describe.schema.json",
    "upload-fit.metadata.json": "upload-metadata.schema.json",
    "put-result.metadata.json": "put-result.schema.json",
}

RESULT_FIXTURES = {
    "capabilities.result.json": "capabilities-result.schema.json",
    "health.result.json": "health-result.schema.json",
    "create-fit.result.json": "create-result.schema.json",
    "create-predict.result.json": "create-result.schema.json",
    "acquire.result.json": "acquire-result.schema.json",
    "status.result.json": "status-result.schema.json",
    "inputs-list.result.json": "inputs-list-result.schema.json",
    "input-close.result.json": "input-close-result.schema.json",
    "outputs-list.result.json": "outputs-list-result.schema.json",
    "cancel.result.json": "cancel-result.schema.json",
    "model-describe.result.json": "model-describe-result.schema.json",
}

ACTION_REQUESTS = {
    "capabilities.request.json": CAPABILITIES_ACTION,
    "health.request.json": HEALTH_ACTION,
    "create-fit.request.json": CREATE_ACTION,
    "create-predict.request.json": CREATE_ACTION,
    "acquire.request.json": ACQUIRE_ACTION,
    "status.request.json": STATUS_ACTION,
    "inputs-list.request.json": INPUTS_LIST_ACTION,
    "input-close.request.json": INPUT_CLOSE_ACTION,
    "outputs-list.request.json": OUTPUTS_LIST_ACTION,
    "cancel.request.json": CANCEL_ACTION,
    "model-describe.request.json": MODEL_DESCRIBE_ACTION,
}

ARROW_FIXTURES = {
    "fit-multi-batch.arrow": (2, 3),
    "predict-multi-batch.arrow": (2, 3),
    "predict-typed-empty.arrow": (0, 0),
    "prediction-output.arrow": (2, 3),
    "prediction-output-typed-empty.arrow": (0, 0),
}


def _read_json(name: str) -> dict:
    return parse_action_body((JSON_ROOT / name).read_bytes())


def test_v4_has_an_exact_closed_action_surface():
    assert ACTIONS == (
        "transformer.v4.capabilities",
        "transformer.v4.health",
        "transformer.v4.job.create",
        "transformer.v4.job.acquire",
        "transformer.v4.job.status",
        "transformer.v4.job.inputs.list",
        "transformer.v4.job.input.close",
        "transformer.v4.job.outputs.list",
        "transformer.v4.job.cancel",
        "transformer.v4.model.describe",
    )


def test_all_json_schemas_are_valid_closed_draft_2020_12_documents():
    schemas = sorted(SCHEMA_ROOT.glob("*.json"))
    assert {path.name for path in schemas} == {
        "acquire-result.schema.json",
        "acquire.schema.json",
        "action-result.schema.json",
        "cancel-result.schema.json",
        "cancel.schema.json",
        "capabilities-result.schema.json",
        "common.schema.json",
        "create-result.schema.json",
        "create.schema.json",
        "health-result.schema.json",
        "input-close-result.schema.json",
        "input-close.schema.json",
        "inputs-list-result.schema.json",
        "inputs-list.schema.json",
        "model-describe-result.schema.json",
        "model-describe.schema.json",
        "objective-config.schema.json",
        "outputs-list-result.schema.json",
        "outputs-list.schema.json",
        "put-result.schema.json",
        "query.schema.json",
        "status-result.schema.json",
        "status.schema.json",
        "upload-metadata.schema.json",
    }
    for path in schemas:
        schema = json.loads(path.read_text(encoding="utf-8"))
        check_contract_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        if path.name == "common.schema.json":
            assert "$defs" in schema
            continue
        assert schema["type"] == "object"
        assert schema.get("additionalProperties", False) is False


def test_golden_json_documents_match_schemas_and_runtime_parser():
    expected_files = (
        set(REQUEST_FIXTURES)
        | set(RESULT_FIXTURES)
        | {"objective-config.fit.json"}
    )
    assert {path.name for path in JSON_ROOT.glob("*.json")} == expected_files

    for fixture_name, schema_name in REQUEST_FIXTURES.items():
        validate_contract_document(
            _read_json(fixture_name),
            read_contract_schema(schema_name),
        )
    for fixture_name, schema_name in RESULT_FIXTURES.items():
        document = _read_json(fixture_name)
        validate_contract_document(document, read_contract_schema(schema_name))
        validate_contract_document(
            document,
            read_contract_schema("action-result.schema.json"),
        )

    for fixture_name, action in ACTION_REQUESTS.items():
        parsed = validate_action_request(action, _read_json(fixture_name))
        assert parsed["request_id"]
    upload = validate_upload_metadata(_read_json("upload-fit.metadata.json"))
    assert upload["fencing_token"] == 2
    assert upload["data_contract_sha256"] == "a" * 64


def test_fit_objective_fixture_pins_the_cross_language_digest():
    fixture = _read_json("objective-config.fit.json")
    validate_contract_document(
        fixture,
        read_contract_schema("objective-config.schema.json"),
    )
    config = TrainConfig(
        lr=0.0005,
        batch_size=256,
        epochs=25,
        loss_stage=4,
        loss_schedule="epoch",
        stage_size=5,
        use_amp=False,
        weight_decay=0.00001,
        direct_loss_weights=(1, 1, 1, 1, 1, 1),
        selection=CheckpointSelectionConfig(min_delta=0, patience=5),
        seed=42,
        deterministic=True,
    )

    assert fixture == objective_config(config)
    assert objective_config_sha256(config) == _read_json(
        "create-fit.request.json"
    )["mlContract"]["objectiveConfigSha256"]


def test_objective_contract_supports_fixed_epoch_selection_mode():
    document = objective_config(TrainConfig())

    validate_contract_document(
        document,
        read_contract_schema("objective-config.schema.json"),
    )
    assert document["selection"] == {
        "enabled": False,
        "aggregation": "global_row_mean",
        "weights": [1.0] * 6,
        "minDelta": 0.0,
        "patience": 0,
        "stagePolicy": "maximum_only_reset",
        "tiePolicy": "earliest",
        "baselinePolicy": "none",
        "invalidScorePolicy": "fail_training",
        "fallbackPolicy": "last_maximum_stage_checkpoint",
    }


def test_fixture_relationships_pin_identity_fence_and_manifest_digest():
    created = _read_json("create-fit.result.json")
    acquired = _read_json("acquire.result.json")
    uploaded = _read_json("put-result.metadata.json")
    listed = _read_json("inputs-list.result.json")
    closed = _read_json("input-close.request.json")

    assert created["jobId"] == acquired["jobId"] == uploaded["jobId"]
    assert created["ownership"]["fencingToken"] == "1"
    assert acquired["ownership"]["fencingToken"] == "2"
    assert listed["items"][0]["commitRevision"] == uploaded["inputRevision"]
    assert canonical_manifest_hash(listed["items"]) == closed["manifestSha256"]

    fit_path = ARROW_ROOT / "fit-multi-batch.arrow"
    assert uploaded["bytes"] == fit_path.stat().st_size
    assert uploaded["sha256"] == hashlib.sha256(fit_path.read_bytes()).hexdigest()
    assert uploaded["schemaFingerprint"] == schema_fingerprint(
        ipc.open_file(fit_path).schema
    )


def test_result_schemas_reject_missing_and_extra_fields():
    for fixture_name, schema_name in RESULT_FIXTURES.items():
        document = _read_json(fixture_name)
        schema = read_contract_schema(schema_name)
        missing = deepcopy(document)
        missing.pop(schema["required"][-1])
        with pytest.raises(ValidationError):
            validate_contract_document(missing, schema)
        with pytest.raises(ValidationError):
            validate_contract_document({**document, "internalPath": "/tmp/x"}, schema)


def test_arrow_v4_fixtures_use_exact_nonnullable_fixed_size_float32_schemas():
    assert {path.name for path in ARROW_ROOT.glob("*.arrow")} == set(
        ARROW_FIXTURES
    )
    for name, (expected_batches, expected_rows) in ARROW_FIXTURES.items():
        reader = ipc.open_file(ARROW_ROOT / name)
        assert reader.num_record_batches == expected_batches
        assert sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        ) == expected_rows
        for field in reader.schema:
            assert field.nullable is False
            assert pa.types.is_fixed_size_list(field.type)
            assert field.type.value_type == pa.float32()
            assert field.type.value_field.name == "item"
            assert field.type.value_field.nullable is True

    fit = ipc.open_file(ARROW_ROOT / "fit-multi-batch.arrow").schema
    predict = ipc.open_file(ARROW_ROOT / "predict-multi-batch.arrow").schema
    output = ipc.open_file(ARROW_ROOT / "prediction-output.arrow").schema
    assert fit.names == ["src", "tgt"]
    assert fit.field("src").type.list_size == 4
    assert fit.field("tgt").type.list_size == 6
    assert predict.names == ["src"]
    assert predict.field("src").type.list_size == 4
    assert output.names == ["out"]
    assert output.field("out").type.list_size == 6


def test_arrow_golden_fixtures_are_reproducible(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.contracts.flight.v4.fixtures.generate_arrow_fixtures",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )
    for name in ARROW_FIXTURES:
        assert (tmp_path / name).read_bytes() == (ARROW_ROOT / name).read_bytes()
