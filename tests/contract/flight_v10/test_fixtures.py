from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy

import pyarrow.ipc as ipc
import pytest
import torch
from jsonschema.exceptions import ValidationError

from app.contracts.flight.v10.arrow import (
    canonical_input_schema,
    canonical_prediction_schema,
)
from app.contracts.flight.v10.fixtures.generate_arrow_fixtures import (
    FEATURE_DIM,
    MISSING_OBSERVATION_SOURCE_ENCODING,
    PRODUCTION_CORE_V2_FEATURE_DIM,
    PRODUCTION_CORE_V2_SOURCE_ENCODING,
    PRODUCTION_CORE_V6_FEATURE_DIM,
    PRODUCTION_CORE_V6_SOURCE_ENCODING,
    SEQ_LEN,
    SOURCE_ENCODING,
)
from app.contracts.indexed_feature_blocks import canonical_source_encoding
from app.contracts.worker.v11.objective import (
    ObjectiveConfig,
    default_objective,
    objective_config,
    objective_config_sha256,
)
from app.project import PROJECT_ROOT
from app.service.adapters.inbound.flight.arrow import (
    InputBatchValidator,
    schema_fingerprint,
)
from app.service.adapters.inbound.flight.constants import (
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
from app.service.adapters.inbound.flight.documents import (
    canonical_manifest_hash,
    parse_action_body,
)
from app.service.adapters.inbound.flight.validation import (
    validate_action_request,
    validate_upload_metadata,
)
from app.service.domain.errors import ServiceError
from app.worker.data.arrow import (
    iter_committed_fit_arrow,
    iter_committed_source_arrow,
)
from tests.support.flight_contract_schema import (
    check_contract_schema,
    read_contract_schema,
    validate_contract_document,
)

FIXTURE_ROOT = (
    PROJECT_ROOT / "app"
    / "contracts"
    / "flight"
    / "v10"
    / "fixtures"
)
JSON_ROOT = FIXTURE_ROOT / "json"
ARROW_ROOT = FIXTURE_ROOT / "arrow"
SOURCE_ENCODING_ROOT = FIXTURE_ROOT / "source-encodings"
SCHEMA_ROOT = FIXTURE_ROOT.parent / "schemas"
NODE_OBJECTIVE_DIGEST = FIXTURE_ROOT / "objective_config_sha256.mjs"

REQUEST_FIXTURES = {
    "capabilities.request.json": "query.schema.json",
    "health.request.json": "query.schema.json",
    "create-fit.request.json": "create.schema.json",
    "create-fit-published-model.request.json": "create.schema.json",
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
    "create-fit-published-model.request.json": CREATE_ACTION,
    "create-predict.request.json": CREATE_ACTION,
    "acquire.request.json": ACQUIRE_ACTION,
    "status.request.json": STATUS_ACTION,
    "inputs-list.request.json": INPUTS_LIST_ACTION,
    "input-close.request.json": INPUT_CLOSE_ACTION,
    "outputs-list.request.json": OUTPUTS_LIST_ACTION,
    "cancel.request.json": CANCEL_ACTION,
    "model-describe.request.json": MODEL_DESCRIBE_ACTION,
}

INPUT_ARROW_FIXTURES = {
    "fit-multi-batch.arrow": (
        "fit", 2, 2, SOURCE_ENCODING, SEQ_LEN, FEATURE_DIM,
    ),
    "predict-multi-batch.arrow": (
        "predict", 2, 2, SOURCE_ENCODING, SEQ_LEN, FEATURE_DIM,
    ),
    "predict-production-core-v2.arrow": (
        "predict", 1, 1, PRODUCTION_CORE_V2_SOURCE_ENCODING, 10,
        PRODUCTION_CORE_V2_FEATURE_DIM,
    ),
    "predict-production-core-v6-hour-boundary.arrow": (
        "predict", 1, 1, PRODUCTION_CORE_V6_SOURCE_ENCODING, 10,
        PRODUCTION_CORE_V6_FEATURE_DIM,
    ),
    "predict-missing-observation.arrow": (
        "predict", 1, 1, MISSING_OBSERVATION_SOURCE_ENCODING, 2, 2,
    ),
    "predict-typed-empty.arrow": (
        "predict", 0, 0, SOURCE_ENCODING, SEQ_LEN, FEATURE_DIM,
    ),
}
INVALID_INPUT_ARROW_FIXTURES = {"predict-missing-native-prefix.arrow"}
OUTPUT_ARROW_FIXTURES = {
    "prediction-output.arrow": (2, 3),
    "prediction-output-typed-empty.arrow": (0, 0),
}
ARROW_FIXTURES = (
    set(INPUT_ARROW_FIXTURES)
    | set(INVALID_INPUT_ARROW_FIXTURES)
    | set(OUTPUT_ARROW_FIXTURES)
)


def _read_json(name: str) -> dict:
    return parse_action_body((JSON_ROOT / name).read_bytes())


def test_v10_has_an_exact_closed_action_surface():
    assert ACTIONS == (
        "transformer.v10.capabilities",
        "transformer.v10.health",
        "transformer.v10.job.create",
        "transformer.v10.job.acquire",
        "transformer.v10.job.status",
        "transformer.v10.job.inputs.list",
        "transformer.v10.job.input.close",
        "transformer.v10.job.outputs.list",
        "transformer.v10.job.cancel",
        "transformer.v10.model.describe",
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
        "diagnostics.schema.json",
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


def test_status_exposes_only_operation_specific_compact_progress():
    schema = read_contract_schema("status-result.schema.json")
    status = _read_json("status.result.json")

    validate_contract_document(status, schema)

    pending_fit = deepcopy(status)
    pending_fit["progress"] = {}
    validate_contract_document(pending_fit, schema)

    predict = deepcopy(status)
    predict["operation"] = "predict"
    predict["initialization"] = None
    predict["resolvedModelRef"] = "mdl_generation"
    predict["progress"] = {"ordinal": 0, "rows": 3}
    validate_contract_document(predict, schema)

    extended = deepcopy(status)
    extended["progress"]["grad_norm"] = 1.0
    with pytest.raises(ValidationError):
        validate_contract_document(extended, schema)


def test_results_preserve_complete_published_model_lineage():
    lineage = {
        "kind": "publishedModel",
        "parentModelRef": "mdl_01j2x5f4x6h7k8m9n0p1q2r3s4",
        "parentCheckpointSha256": "e" * 64,
        "parentDataContractSha256": "4" * 64,
        "dataContractSha256": "2" * 64,
    }
    description = deepcopy(_read_json("model-describe.result.json"))
    description["initialization"] = lineage
    description["dataContract"]["dataContractSha256"] = "2" * 64
    validate_contract_document(
        description,
        read_contract_schema("model-describe-result.schema.json"),
    )

    status = deepcopy(_read_json("status.result.json"))
    status["initialization"] = lineage
    status["dataContract"]["dataContractSha256"] = "2" * 64
    status["results"]["checkpoint"]["initialization"] = lineage
    status["results"]["checkpoint"]["dataContract"][
        "dataContractSha256"
    ] = "2" * 64
    validate_contract_document(
        status,
        read_contract_schema("status-result.schema.json"),
    )

    incomplete = deepcopy(description)
    del incomplete["initialization"]["parentDataContractSha256"]
    with pytest.raises(ValidationError):
        validate_contract_document(
            incomplete,
            read_contract_schema("model-describe-result.schema.json"),
        )


def test_fit_objective_fixture_pins_the_cross_language_digest():
    fixture = _read_json("objective-config.fit.json")
    validate_contract_document(
        fixture,
        read_contract_schema("objective-config.schema.json"),
    )
    request = _read_json("create-fit.request.json")
    config = ObjectiveConfig.from_document({
        "targets": request["targets"],
        "objective": fixture,
    })
    parsed = validate_action_request(CREATE_ACTION, request)

    assert objective_config(config) == {
        "targets": request["targets"],
        "objective": fixture,
    }
    assert objective_config_sha256(config) == parsed["ml_contract"][
        "objectiveConfigSha256"
    ]

    legacy_identity = deepcopy(fixture)
    legacy_identity["directLosses"][0]["semantic"] = "meanReturn"
    with pytest.raises(ValidationError):
        validate_contract_document(
            legacy_identity,
            read_contract_schema("objective-config.schema.json"),
        )


def test_model_description_returns_profile_without_duplicate_target_list():
    description = _read_json("model-describe.result.json")
    capabilities = _read_json("capabilities.result.json")

    assert description["dataContract"]["profile"] == (
        "research-dividend-events-v2"
    )
    assert description["mlContract"]["targetSchemaId"] == "inventory.target.v2"
    assert description["mlContract"]["targetWidth"] == 6
    assert "targets" not in description
    assert "profile" not in capabilities["mlContract"]
    assert "profiles" not in capabilities["mlContract"]


def test_node_jcs_matches_the_normative_objective_digest():
    node = shutil.which("node")
    assert node is not None, (
        "Node.js is required for the cross-language contract test"
    )

    result = subprocess.run(
        [
            node,
            str(NODE_OBJECTIVE_DIGEST),
            str(JSON_ROOT / "objective-config.fit.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stderr == ""
    request = _read_json("create-fit.request.json")
    parsed = validate_action_request(CREATE_ACTION, request)
    assert result.stdout.strip() == parsed["ml_contract"][
        "objectiveConfigSha256"
    ]


def test_objective_contract_activates_all_declared_losses_without_a_schedule():
    config = default_objective()
    document = config.objective

    validate_contract_document(
        document,
        read_contract_schema("objective-config.schema.json"),
    )
    assert document["aggregation"] == "WeightedSum"
    assert document["balancing"] == {"operator": "Static"}
    assert len(document["directLosses"]) == len(config.targets)
    assert "stagePolicy" not in document


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


def test_arrow_v10_fixtures_use_exact_compact_and_prediction_schemas():
    assert {path.name for path in ARROW_ROOT.glob("*.arrow")} == ARROW_FIXTURES
    for name, (
        operation,
        expected_batches,
        expected_chunks,
        source_encoding,
        seq_len,
        feature_dim,
    ) in (
        INPUT_ARROW_FIXTURES.items()
    ):
        reader = ipc.open_file(ARROW_ROOT / name)
        assert reader.num_record_batches == expected_batches
        assert sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        ) == expected_chunks
        assert reader.schema == canonical_input_schema(
            operation,
            source_encoding,
            seq_len,
            feature_dim,
        )

    for name, (expected_batches, expected_rows) in (
        OUTPUT_ARROW_FIXTURES.items()
    ):
        reader = ipc.open_file(ARROW_ROOT / name)
        assert reader.num_record_batches == expected_batches
        assert sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        ) == expected_rows
        assert reader.schema == canonical_prediction_schema("out")


def test_cross_project_source_encoding_fixtures_pin_profile_geometries():
    assert {path.name for path in SOURCE_ENCODING_ROOT.glob("*.json")} == {
        "production-core-v2.json",
        "production-core-v6.json",
    }
    cases = (
        (
            "production-core-v2.json",
            PRODUCTION_CORE_V2_SOURCE_ENCODING,
            PRODUCTION_CORE_V2_FEATURE_DIM,
        ),
        (
            "production-core-v6.json",
            PRODUCTION_CORE_V6_SOURCE_ENCODING,
            PRODUCTION_CORE_V6_FEATURE_DIM,
        ),
    )
    for name, expected, feature_dim in cases:
        document = json.loads(
            (SOURCE_ENCODING_ROOT / name).read_text(encoding="utf-8")
        )
        assert canonical_source_encoding(
            document,
            feature_dim=feature_dim,
        ) == expected


def test_compact_fixtures_reconstruct_the_normative_dense_tensors():
    fit = list(iter_committed_fit_arrow(
        str(ARROW_ROOT / "fit-multi-batch.arrow"),
        expected_rows=3,
        expected_chunks=2,
        expected_native_rows=(7, 5),
        source_encoding=SOURCE_ENCODING,
        seq_len=SEQ_LEN,
        feature_dim=FEATURE_DIM,
        targets=default_objective().targets,
    ))
    features = torch.cat([batch.features for batch in fit])
    targets = torch.cat([batch.targets for batch in fit])

    assert features.shape == (3, 2, 4)
    torch.testing.assert_close(
        features[0, 0],
        torch.tensor([0.1, float("nan"), 10, 11]),
        equal_nan=True,
    )
    torch.testing.assert_close(features[2], torch.tensor([
        [-1, -2, -10, -11],
        [-2, -3, -12, -13],
    ], dtype=torch.float32))
    assert targets.shape == (3, 6)
    fit_reader = ipc.open_file(ARROW_ROOT / "fit-multi-batch.arrow")
    range_index = fit_reader.schema.get_field_index("rangeOrdinal")
    offset_index = fit_reader.schema.get_field_index("exampleOffset")
    assert [
        fit_reader.get_batch(index).column(range_index).to_pylist()
        for index in range(fit_reader.num_record_batches)
    ] == [[0], [0]]
    assert [
        fit_reader.get_batch(index).column(offset_index).to_pylist()
        for index in range(fit_reader.num_record_batches)
    ] == [[0], [2]]

    predict = list(iter_committed_source_arrow(
        str(ARROW_ROOT / "predict-multi-batch.arrow"),
        expected_rows=3,
        expected_chunks=2,
        expected_native_rows=(7, 5),
        source_encoding=SOURCE_ENCODING,
        seq_len=SEQ_LEN,
        feature_dim=FEATURE_DIM,
    ))
    assert torch.cat(predict).shape == (3, 2, 4)


def test_profile_fixtures_reconstruct_one_and_multi_timeframe_blocks():
    v2 = list(iter_committed_source_arrow(
        str(ARROW_ROOT / "predict-production-core-v2.arrow"),
        expected_rows=1,
        expected_chunks=1,
        expected_native_rows=(109,),
        source_encoding=PRODUCTION_CORE_V2_SOURCE_ENCODING,
        seq_len=10,
        feature_dim=PRODUCTION_CORE_V2_FEATURE_DIM,
    ))
    assert v2[0].shape == (1, 10, PRODUCTION_CORE_V2_FEATURE_DIM)
    assert v2[0][0, 0, 0].item() == 0.0
    assert v2[0][0, 0, -1].item() == 99.0
    assert v2[0][0, 9, 0].item() == 9.0
    assert v2[0][0, 9, -1].item() == 108.0

    v6 = list(iter_committed_source_arrow(
        str(ARROW_ROOT / "predict-production-core-v6-hour-boundary.arrow"),
        expected_rows=1,
        expected_chunks=1,
        expected_native_rows=(109, 10),
        source_encoding=PRODUCTION_CORE_V6_SOURCE_ENCODING,
        seq_len=10,
        feature_dim=PRODUCTION_CORE_V6_FEATURE_DIM,
    ))
    assert v6[0].shape == (1, 10, PRODUCTION_CORE_V6_FEATURE_DIM)
    hourly_position = 64_800
    assert v6[0][0, 5, hourly_position].item() == 10_000.0
    assert v6[0][0, 6, hourly_position].item() == 10_001.0
    assert v6[0][0, 9, hourly_position].item() == 10_002.0


def test_missing_observation_fixture_uses_explicit_local_offsets():
    batches = list(iter_committed_source_arrow(
        str(ARROW_ROOT / "predict-missing-observation.arrow"),
        expected_rows=1,
        expected_chunks=1,
        expected_native_rows=(5,),
        source_encoding=MISSING_OBSERVATION_SOURCE_ENCODING,
        seq_len=2,
        feature_dim=2,
    ))

    torch.testing.assert_close(
        batches[0],
        torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
    )


def test_missing_native_prefix_fixture_is_rejected_before_commit():
    path = ARROW_ROOT / "predict-missing-native-prefix.arrow"
    reader = ipc.open_file(path)
    validator = InputBatchValidator(
        "predict",
        reader.schema,
        source_encoding=MISSING_OBSERVATION_SOURCE_ENCODING,
        targets=default_objective().targets,
        seq_len=2,
        expected_feature_dim=2,
        max_batch_bytes=path.stat().st_size,
        max_payload_bytes=path.stat().st_size,
        max_rows=1,
    )

    with pytest.raises(ServiceError, match="observation offset is outside"):
        validator.validate_batch(reader.get_batch(0))


def test_arrow_golden_fixtures_are_reproducible(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.contracts.flight.v10.fixtures.generate_arrow_fixtures",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )
    for name in ARROW_FIXTURES:
        assert (tmp_path / name).read_bytes() == (ARROW_ROOT / name).read_bytes()
