from copy import deepcopy
from datetime import datetime
import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import uuid

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from app.data.arrow import (
    empty_predictions_table,
    predictions_to_table,
    table_to_source_tensor,
    table_to_tensors,
    validate_arrow_table,
)
from app.flight.constants import (
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CREATE_ACTION,
    HEALTH_ACTION,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
)
from app.flight.contract import (
    parse_action_body,
    validate_action_request,
    validate_upload_metadata,
)
from app.flight.arrow import schema_fingerprint


FIXTURE_ROOT = Path(__file__).parents[1] / "contracts" / "flight" / "v1" / "fixtures"
JSON_ROOT = FIXTURE_ROOT / "json"
ARROW_ROOT = FIXTURE_ROOT / "arrow"
GENERATOR = FIXTURE_ROOT / "generate_arrow_fixtures.py"
SCHEMA_ROOT = FIXTURE_ROOT.parent / "schemas"

RESULT_FIXTURES = {
    "capabilities.result.json": "capabilities-result.schema.json",
    "health.result.json": "health-result.schema.json",
    "create-fit.result.json": "create-result.schema.json",
    "seal.result.json": "seal-result.schema.json",
    "start.result.json": "start-result.schema.json",
    "cancel.result.json": "cancel-result.schema.json",
    "status.result.json": "status-result.schema.json",
}

RESULT_REQUIRED_FIELDS = {
    "capabilities-result.schema.json": {
        "contract", "version", "requestId", "protocolVersions", "service",
        "schemaIds", "limits", "devices", "queue", "supportedOperations",
        "features",
    },
    "health-result.schema.json": {
        "contract", "version", "requestId", "live", "ready", "draining",
        "ledger", "cuda", "disk", "metrics",
    },
    "create-result.schema.json": {
        "contract", "version", "requestId", "jobId", "operation", "state",
        "revision", "device", "resolvedModelRef", "limits", "upload",
    },
    "seal-result.schema.json": {
        "contract", "version", "requestId", "jobId", "state", "revision",
        "manifestSha256", "inputs",
    },
    "start-result.schema.json": {
        "contract", "version", "requestId", "jobId", "state", "revision",
        "device",
    },
    "cancel-result.schema.json": {
        "contract", "version", "requestId", "jobId", "state", "revision",
    },
    "status-result.schema.json": {
        "contract", "version", "requestId", "jobId", "operation", "state",
        "revision", "timestamps", "device", "committedInputs", "progress",
        "attempt", "error", "results", "pollAfterMs",
    },
}

ARROW_FIXTURES = {
    "fit-multi-batch.arrow": (2, 3),
    "predict-multi-batch.arrow": (2, 3),
    "predict-typed-empty.arrow": (0, 0),
    "prediction-output.arrow": (2, 3),
    "prediction-output-typed-empty.arrow": (0, 0),
}


def _read_json(name: str) -> dict:
    path = JSON_ROOT / name
    return parse_action_body(path.read_bytes())


def _open_fixture(name: str):
    return ipc.open_file(ARROW_ROOT / name)


def _read_schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text())


def _schema_property(schema: dict, *path: str) -> dict:
    for name in path:
        schema = schema["properties"][name]
    return schema


def _json_equal(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _matches_type(value, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise AssertionError(f"unsupported JSON Schema type in contract test: {expected}")


def _validate_schema_subset(value, schema: dict, path: str = "$") -> None:
    """Validate the contract's deliberately small Draft 2020-12 subset.

    This keeps golden-contract verification dependency-free while exercising the
    required, closed-object and conditional semantics used by result schemas.
    It is not intended to be a general JSON Schema implementation.
    """

    assert "$ref" not in schema, f"{path}: direct result schemas must be self-contained"

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        assert any(_matches_type(value, expected) for expected in expected_types), (
            f"{path}: expected type {expected_types}, got {type(value).__name__}"
        )

    if "const" in schema:
        assert _json_equal(value, schema["const"]), (
            f"{path}: expected const {schema['const']!r}, got {value!r}"
        )
    if "enum" in schema:
        assert any(_json_equal(value, candidate) for candidate in schema["enum"]), (
            f"{path}: {value!r} is not in {schema['enum']!r}"
        )

    if isinstance(value, dict):
        required = set(schema.get("required", ()))
        missing = required - value.keys()
        assert not missing, f"{path}: missing required fields {sorted(missing)!r}"

        properties = schema.get("properties", {})
        unknown = value.keys() - properties.keys()
        additional = schema.get("additionalProperties", True)
        if additional is False:
            assert not unknown, f"{path}: unexpected fields {sorted(unknown)!r}"
        elif isinstance(additional, dict):
            for key in unknown:
                _validate_schema_subset(value[key], additional, f"{path}.{key}")

        for key, property_schema in properties.items():
            if key in value:
                _validate_schema_subset(value[key], property_schema, f"{path}.{key}")

    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{path}: too few items"
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], f"{path}: too many items"
        if schema.get("uniqueItems"):
            canonical = [json.dumps(item, sort_keys=True) for item in value]
            assert len(canonical) == len(set(canonical)), f"{path}: duplicate items"

        prefix = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix[:len(value)]):
            _validate_schema_subset(value[index], item_schema, f"{path}[{index}]")
        items = schema.get("items")
        if items is False:
            assert len(value) <= len(prefix), f"{path}: items beyond prefix are forbidden"
        elif isinstance(items, dict):
            start = len(prefix) if prefix else 0
            for index, item in enumerate(value[start:], start=start):
                _validate_schema_subset(item, items, f"{path}[{index}]")

    if isinstance(value, str):
        if "minLength" in schema:
            assert len(value) >= schema["minLength"], f"{path}: string is too short"
        if "maxLength" in schema:
            assert len(value) <= schema["maxLength"], f"{path}: string is too long"
        if "pattern" in schema:
            assert re.search(schema["pattern"], value), f"{path}: pattern mismatch"
        if schema.get("format") == "uuid":
            assert str(uuid.UUID(value)) == value.lower(), f"{path}: invalid UUID"
        elif schema.get("format") == "date-time":
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            assert parsed.tzinfo is not None, f"{path}: date-time must include timezone"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema:
            assert value >= schema["minimum"], f"{path}: below minimum"
        if "maximum" in schema:
            assert value <= schema["maximum"], f"{path}: above maximum"
        if "exclusiveMinimum" in schema:
            assert value > schema["exclusiveMinimum"], f"{path}: below exclusiveMinimum"
        if "exclusiveMaximum" in schema:
            assert value < schema["exclusiveMaximum"], f"{path}: above exclusiveMaximum"

    for branch in schema.get("allOf", ()):
        condition = branch.get("if")
        if condition is None:
            _validate_schema_subset(value, branch, path)
            continue
        try:
            _validate_schema_subset(value, condition, path)
        except AssertionError:
            selected = branch.get("else")
        else:
            selected = branch.get("then")
        if selected is not None:
            _validate_schema_subset(value, selected, path)


def test_json_golden_fixtures_match_runtime_contract():
    assert {path.name for path in JSON_ROOT.glob("*.json")} == {
        "cancel.request.json",
        "cancel.result.json",
        "capabilities.request.json",
        "capabilities.result.json",
        "create-fit.request.json",
        "create-fit.result.json",
        "create-predict.request.json",
        "health.request.json",
        "health.result.json",
        "put-result.metadata.json",
        "seal.request.json",
        "seal.result.json",
        "start.request.json",
        "start.result.json",
        "status.request.json",
        "status.result.json",
        "upload-fit.metadata.json",
    }

    fit = validate_action_request(
        CREATE_ACTION,
        _read_json("create-fit.request.json"),
    )
    assert fit["operation"] == "fit"
    assert fit["model_config"].seq_len == 2

    predict = validate_action_request(
        CREATE_ACTION,
        _read_json("create-predict.request.json"),
    )
    assert predict["operation"] == "predict"
    assert predict["model_selector"] == "modelRef"

    seal = validate_action_request(SEAL_ACTION, _read_json("seal.request.json"))
    assert [item["ordinal"] for item in seal["manifest"]] == [0]

    upload = validate_upload_metadata(_read_json("upload-fit.metadata.json"))
    assert upload["ordinal"] == 0
    assert upload["rows"] == 3

    put_result = _read_json("put-result.metadata.json")
    assert put_result["status"] == "committed"
    assert put_result["batches"] == 2
    assert len(put_result["sha256"]) == 64
    assert len(put_result["schemaFingerprint"]) == 64

    for action, name in (
        (CAPABILITIES_ACTION, "capabilities.request.json"),
        (HEALTH_ACTION, "health.request.json"),
        (STATUS_ACTION, "status.request.json"),
        (START_ACTION, "start.request.json"),
        (CANCEL_ACTION, "cancel.request.json"),
    ):
        assert validate_action_request(action, _read_json(name))["request_id"]

    result = _read_json("create-fit.result.json")
    assert result["state"] == "UPLOADING"
    assert result["upload"]["oneDoPutIsOneSemanticFrame"] is True
    assert result["limits"]["maxRowsPerPayload"] == 2_000_000

    for request_name, result_name in (
        ("capabilities.request.json", "capabilities.result.json"),
        ("health.request.json", "health.result.json"),
        ("create-fit.request.json", "create-fit.result.json"),
        ("seal.request.json", "seal.result.json"),
        ("start.request.json", "start.result.json"),
        ("cancel.request.json", "cancel.result.json"),
        ("status.request.json", "status.result.json"),
    ):
        assert _read_json(result_name)["requestId"] == _read_json(request_name)["requestId"]

    assert result["upload"]["descriptorPath"][3] == result["jobId"]
    assert _read_json("seal.result.json")["jobId"] == result["jobId"]
    assert _read_json("start.result.json")["jobId"] == result["jobId"]
    assert _read_json("cancel.result.json")["jobId"] == result["jobId"]

    manifest = _read_json("seal.request.json")["manifest"]
    canonical_manifest = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert _read_json("seal.result.json")["manifestSha256"] == hashlib.sha256(
        canonical_manifest
    ).hexdigest()

    fit_path = ARROW_ROOT / "fit-multi-batch.arrow"
    fit_reader = ipc.open_file(fit_path)
    assert upload["rows"] == sum(
        fit_reader.get_batch(index).num_rows
        for index in range(fit_reader.num_record_batches)
    )
    assert put_result["bytes"] == fit_path.stat().st_size
    assert put_result["sha256"] == hashlib.sha256(fit_path.read_bytes()).hexdigest()
    assert put_result["schemaFingerprint"] == schema_fingerprint(fit_reader.schema)
    assert seal["manifest"][0]["sha256"] == put_result["sha256"]
    assert fit_reader.get_batch(0).column(0).value_lengths()[0].as_py() % fit["model_config"].seq_len == 0

    status_input = _read_json("status.result.json")["committedInputs"][0]
    for key in ("rows", "batches", "bytes", "sha256", "schemaFingerprint"):
        assert status_input[key] == put_result[key]


def test_language_neutral_json_schemas_are_parseable_and_strict_at_boundaries():
    schemas = {path.name: json.loads(path.read_text()) for path in SCHEMA_ROOT.glob("*.json")}
    assert {
        "common.schema.json",
        "create.schema.json",
        "query.schema.json",
        "status.schema.json",
        "job-mutation.schema.json",
        "seal.schema.json",
        "upload-metadata.schema.json",
        "put-result.schema.json",
        "action-result.schema.json",
        "capabilities-result.schema.json",
        "health-result.schema.json",
        "create-result.schema.json",
        "seal-result.schema.json",
        "start-result.schema.json",
        "cancel-result.schema.json",
        "status-result.schema.json",
    } <= set(schemas)
    for document in schemas.values():
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document["type"] == "object"
    for name in (
        "create.schema.json",
        "query.schema.json",
        "status.schema.json",
        "job-mutation.schema.json",
        "seal.schema.json",
        "upload-metadata.schema.json",
        "put-result.schema.json",
        *RESULT_FIXTURES.values(),
    ):
        assert schemas[name]["additionalProperties"] is False

    for name, required in RESULT_REQUIRED_FIELDS.items():
        assert set(schemas[name]["required"]) == required

    closed_nested_objects = {
        "capabilities-result.schema.json": (
            ("service",),
            ("schemaIds",),
            ("limits",),
            ("devices",),
            ("devices", "cpu"),
            ("devices", "cuda"),
            ("queue",),
            ("features",),
        ),
        "health-result.schema.json": (
            ("ledger",),
            ("cuda",),
            ("disk",),
            ("metrics",),
        ),
        "create-result.schema.json": (("device",), ("limits",), ("upload",)),
        "start-result.schema.json": (("device",),),
        "status-result.schema.json": (
            ("timestamps",),
            ("device",),
            ("error",),
            ("results",),
            ("results", "checkpoint"),
            ("results", "checkpoint", "modelConfig"),
            ("results", "checkpoint", "trainConfig"),
            ("results", "checkpoint", "dataSchema"),
            ("results", "checkpoint", "checkpointSelection"),
        ),
    }
    for name, paths in closed_nested_objects.items():
        for path in paths:
            assert _schema_property(schemas[name], *path)["additionalProperties"] is False

    checkpoint = _schema_property(
        schemas["status-result.schema.json"], "results", "checkpoint"
    )
    assert set(checkpoint["required"]) == {
        "format",
        "serviceVersion",
        "sha256",
        "bytes",
        "modelConfig",
        "trainConfig",
        "dataSchema",
        "checkpointSelection",
    }

    assert {
        branch["$ref"] for branch in schemas["action-result.schema.json"]["oneOf"]
    } == set(RESULT_FIXTURES.values())


def test_result_golden_fixtures_match_specific_language_neutral_schemas():
    for fixture_name, schema_name in RESULT_FIXTURES.items():
        _validate_schema_subset(_read_json(fixture_name), _read_schema(schema_name))


def test_result_schemas_enforce_required_and_closed_object_semantics():
    for fixture_name, schema_name in RESULT_FIXTURES.items():
        fixture = _read_json(fixture_name)
        schema = _read_schema(schema_name)

        missing_required = deepcopy(fixture)
        missing_required.pop(schema["required"][-1])
        with pytest.raises(AssertionError, match="missing required"):
            _validate_schema_subset(missing_required, schema)

        extra_top_level = {**fixture, "checkpointPath": "/srv/models/model.pth"}
        with pytest.raises(AssertionError, match="unexpected fields"):
            _validate_schema_subset(extra_top_level, schema)

    status = _read_json("status.result.json")
    missing_checkpoint_metadata = deepcopy(status)
    missing_checkpoint_metadata["results"]["checkpoint"].pop("dataSchema")
    with pytest.raises(AssertionError, match="missing required"):
        _validate_schema_subset(
            missing_checkpoint_metadata,
            _read_schema("status-result.schema.json"),
        )

    leaked_checkpoint_path = deepcopy(status)
    leaked_checkpoint_path["results"]["checkpoint"]["checkpointPath"] = (
        "/srv/transformer/state/models/checkpoint.pth"
    )
    with pytest.raises(AssertionError, match="unexpected fields"):
        _validate_schema_subset(
            leaked_checkpoint_path,
            _read_schema("status-result.schema.json"),
        )


def test_result_schemas_enforce_device_state_and_result_conditionals():
    health_schema = _read_schema("health-result.schema.json")
    unhealthy_ready = deepcopy(_read_json("health.result.json"))
    unhealthy_ready["draining"] = True
    with pytest.raises(AssertionError, match="expected const False"):
        _validate_schema_subset(unhealthy_ready, health_schema)

    capabilities_schema = _read_schema("capabilities-result.schema.json")
    impossible_cuda = deepcopy(_read_json("capabilities.result.json"))
    impossible_cuda["devices"]["cuda"]["deviceCount"] = 1
    with pytest.raises(AssertionError, match="expected const 0"):
        _validate_schema_subset(impossible_cuda, capabilities_schema)

    create_schema = _read_schema("create-result.schema.json")
    fit_with_resolved_model = deepcopy(_read_json("create-fit.result.json"))
    fit_with_resolved_model["resolvedModelRef"] = "mdl_should_not_exist_for_fit"
    with pytest.raises(AssertionError, match="expected const None"):
        _validate_schema_subset(fit_with_resolved_model, create_schema)

    start_schema = _read_schema("start-result.schema.json")
    cuda_fell_back_to_cpu = deepcopy(_read_json("start.result.json"))
    cuda_fell_back_to_cpu["device"] = {"requested": "cuda", "selected": "cpu"}
    with pytest.raises(AssertionError, match="expected const 'cuda'"):
        _validate_schema_subset(cuda_fell_back_to_cpu, start_schema)

    status_schema = _read_schema("status-result.schema.json")
    successful_fit_without_model = deepcopy(_read_json("status.result.json"))
    successful_fit_without_model["results"]["modelRef"] = None
    with pytest.raises(AssertionError, match=r"expected type \['string'\]"):
        _validate_schema_subset(successful_fit_without_model, status_schema)

    nonterminal = deepcopy(_read_json("status.result.json"))
    nonterminal.update({"state": "RUNNING", "pollAfterMs": 100, "error": None})
    nonterminal["results"] = {
        "outputs": [],
        "modelRef": None,
        "checkpoint": None,
    }
    _validate_schema_subset(nonterminal, status_schema)

    queued = deepcopy(nonterminal)
    queued.update({"state": "QUEUED", "attempt": 0, "pollAfterMs": 100})
    _validate_schema_subset(queued, status_schema)

    nonterminal["pollAfterMs"] = 0
    with pytest.raises(AssertionError, match="below minimum"):
        _validate_schema_subset(nonterminal, status_schema)

    failed_without_error = deepcopy(nonterminal)
    failed_without_error.update({"state": "FAILED", "pollAfterMs": 0})
    with pytest.raises(AssertionError, match=r"expected type \['object'\]"):
        _validate_schema_subset(failed_without_error, status_schema)

    cancelled = deepcopy(nonterminal)
    cancelled.update({"state": "CANCELLED", "pollAfterMs": 0})
    _validate_schema_subset(cancelled, status_schema)

    cancelled_with_error = deepcopy(cancelled)
    cancelled_with_error["error"] = {
        "code": "CANCELLED",
        "message": "job was cancelled",
    }
    with pytest.raises(AssertionError, match="expected const None"):
        _validate_schema_subset(cancelled_with_error, status_schema)

    wrong_missing_flags = deepcopy(_read_json("status.result.json"))
    wrong_missing_flags["results"]["checkpoint"]["dataSchema"]["missing"][
        "flags"
    ] = "none"
    with pytest.raises(AssertionError, match="expected const 'per-feature'"):
        _validate_schema_subset(wrong_missing_flags, status_schema)


def test_arrow_golden_fixtures_have_expected_batch_and_row_boundaries():
    assert {path.name for path in ARROW_ROOT.glob("*.arrow")} == set(ARROW_FIXTURES)

    for name, (expected_batches, expected_rows) in ARROW_FIXTURES.items():
        reader = _open_fixture(name)
        assert reader.num_record_batches == expected_batches
        assert sum(reader.get_batch(i).num_rows for i in range(expected_batches)) == expected_rows


def test_fit_multi_batch_fixture_uses_current_training_validation():
    reader = _open_fixture("fit-multi-batch.arrow")
    table = reader.read_all()

    validate_arrow_table(table, require_target=True)
    source, target = table_to_tensors(table)

    assert source.shape == (3, 4)
    assert target.shape == (3, 6)
    assert torch.isnan(source[0, 1])
    assert pa.types.is_list(reader.schema.field("src").type)
    assert pa.types.is_large_list(reader.schema.field("tgt").type)


def test_predict_multi_batch_and_typed_empty_fixtures_use_current_validation():
    reader = _open_fixture("predict-multi-batch.arrow")
    table = reader.read_all()
    validate_arrow_table(table, require_target=False)
    source = table_to_source_tensor(table)

    assert source.shape == (3, 4)
    assert pa.types.is_fixed_size_list(reader.schema.field("src").type)

    empty_reader = _open_fixture("predict-typed-empty.arrow")
    empty = empty_reader.read_all()
    validate_arrow_table(empty, require_target=False)
    assert table_to_source_tensor(empty).shape == (0, 0)
    assert empty.schema.field("src").type == pa.large_list(pa.float64())


def test_prediction_output_fixtures_use_current_prediction_validation():
    reader = _open_fixture("prediction-output.arrow")
    table = reader.read_all()
    assert table.schema == pa.schema([
        pa.field("out", pa.list_(pa.float32())),
    ])

    values = torch.tensor(table.column("out").to_pylist(), dtype=torch.float32)
    validated = predictions_to_table(values, "out", expected_rows=table.num_rows)
    assert validated.combine_chunks().column("out") == table.combine_chunks().column("out")

    empty = _open_fixture("prediction-output-typed-empty.arrow").read_all()
    assert empty.equals(empty_predictions_table("out"))


def test_arrow_golden_fixtures_are_reproducible(tmp_path):
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(tmp_path)],
        check=True,
    )

    for name in ARROW_FIXTURES:
        assert (tmp_path / name).read_bytes() == (ARROW_ROOT / name).read_bytes()
