from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import rfc8785
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from app.project import PROJECT_ROOT

SEMANTIC_ROOT = PROJECT_ROOT / "app" / "contracts" / "semantic" / "v1"
FLIGHT_ROOT = PROJECT_ROOT / "app" / "contracts" / "flight" / "v11"
WORKER_ROOT = PROJECT_ROOT / "app" / "contracts" / "worker" / "v12"
CHECKPOINT_ROOT = PROJECT_ROOT / "app" / "contracts" / "checkpoint" / "v6"
METRICS_ROOT = PROJECT_ROOT / "app" / "contracts" / "metrics" / "v5"
FIT_RUN_ROOT = PROJECT_ROOT / "app" / "contracts" / "metrics" / "fit_run" / "v5"

SCHEMA_ROOTS = (
    SEMANTIC_ROOT / "schemas",
    FLIGHT_ROOT / "schemas",
    WORKER_ROOT / "schemas",
    CHECKPOINT_ROOT / "schemas",
    METRICS_ROOT,
    FIT_RUN_ROOT,
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
JOB_ID = "22222222-2222-4222-8222-222222222222"
ATTEMPT_ID = "33333333-3333-4333-8333-333333333333"
SEMANTIC_FIXTURES = (
    "single-regression.json",
    "single-positive-regression.json",
    "single-probability.json",
    "multi-target-shared-resource.json",
    "target-reorder-a-b.json",
    "target-reorder-b-a.json",
    "new-opaque-target.json",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(path for root in SCHEMA_ROOTS for path in root.rglob("*.schema.json"))
    )


@pytest.fixture(scope="module")
def schema_registry() -> Registry:
    resources = []
    for path in _schema_paths():
        schema = _read(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))

    return Registry().with_resources(resources)


def _validate(document: object, schema_path: Path, registry: Registry) -> None:
    Draft202012Validator(
        _read(schema_path),
        registry=registry,
        format_checker=FormatChecker(),
    ).validate(document)


def _fixture(name: str) -> dict:
    return _read(SEMANTIC_ROOT / "fixtures" / name)


def _sha256_jcs(value: object) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _d1_preimages(model_contract: dict) -> dict[str, object]:
    revision = model_contract["objectiveLanguageRevision"]
    target_preimage = {
        "objectiveLanguageRevision": revision,
        "targetContract": model_contract["targetContract"],
    }
    objective_preimage = {
        "objectiveLanguageRevision": revision,
        "objective": model_contract["objective"],
    }
    target = _sha256_jcs(target_preimage)
    objective = _sha256_jcs(objective_preimage)

    return {
        "targetContract": target_preimage,
        "objective": objective_preimage,
        "modelContract": {
            "objectiveLanguageRevision": revision,
            "modelConfig": model_contract["modelConfig"],
            "targetContractSha256": target,
            "objectiveSha256": objective,
        },
    }


def _d1(model_contract: dict) -> dict[str, str]:
    preimages = _d1_preimages(model_contract)

    return {
        "targetContractSha256": _sha256_jcs(preimages["targetContract"]),
        "objectiveSha256": _sha256_jcs(preimages["objective"]),
        "modelContractSha256": _sha256_jcs(preimages["modelContract"]),
    }


def _semantic_reason(
    model_contract: dict,
    *,
    available_direct: set[str] | None = None,
    available_auxiliary: set[str] | None = None,
) -> str | None:
    slots = model_contract["targetContract"]["slots"]
    objective = model_contract["objective"]
    slot_ids = [slot["identity"] for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        return "INVALID_TARGET_CONTRACT"

    for slot in slots:
        constraint = slot["observedConstraint"]
        public = slot["publicPredictionTransformation"]["kind"]
        if constraint["kind"] != "ClosedInterval":
            continue
        minimum = constraint["minimum"]
        maximum = constraint["maximum"]
        if minimum > maximum:
            return "INVALID_TARGET_CONTRACT"
        if public == "Identity":
            return "INVALID_TARGET_CONTRACT"
        if public == "Tanh" and not (minimum <= -1 and maximum >= 1):
            return "INVALID_TARGET_CONTRACT"
        if public == "Sigmoid" and not (minimum <= 0 and maximum >= 1):
            return "INVALID_TARGET_CONTRACT"

    slot_by_id = {slot["identity"]: slot for slot in slots}
    resources = objective["resources"]
    resource_ids = [resource["identity"] for resource in resources]
    if resource_ids != sorted(resource_ids) or len(resource_ids) != len(
        set(resource_ids)
    ):
        return "INVALID_RESOURCE_GRAPH"

    direct = objective["directComponents"]
    auxiliary = objective["auxiliaryComponents"]
    if [item["identity"] for item in auxiliary] != sorted(
        item["identity"] for item in auxiliary
    ):
        return "INVALID_OBJECTIVE"
    component_ids = [item["identity"] for item in direct + auxiliary]
    if len(component_ids) != len(set(component_ids)):
        return "INVALID_OBJECTIVE"

    known_direct = {
        "SmoothL1",
        "BinaryCrossEntropyWithLogits",
        "LogMSE",
    }
    known_auxiliary = {
        "GaussianNLL",
        "ExpectedValue",
        "RiskAdjustedExpectedValue",
    }
    for index, component in enumerate(direct):
        operator = component["operator"]
        if operator not in known_direct:
            return "UNKNOWN_PRIMITIVE"
        if available_direct is not None and operator not in available_direct:
            return "PRIMITIVE_UNAVAILABLE"
        target_role = (
            "logit" if operator == "BinaryCrossEntropyWithLogits" else "estimate"
        )
        observed_role = (
            "probability" if operator == "BinaryCrossEntropyWithLogits" else "observed"
        )
        if index >= len(slots):
            return "INVALID_OBJECTIVE"
        if component["roles"][target_role]["identity"] != slot_ids[index]:
            return "INVALID_OBJECTIVE"
        if component["roles"][observed_role]["identity"] != slot_ids[index]:
            return "INVALID_OBJECTIVE"

        slot = slots[index]
        constraint = slot["observedConstraint"]
        loss_transformation = slot["lossInputTransformation"]["kind"]
        if operator == "BinaryCrossEntropyWithLogits":
            if loss_transformation != "Identity":
                return "INVALID_OBJECTIVE"
            if not (
                constraint["kind"] == "ClosedInterval"
                and constraint["minimum"] >= 0
                and constraint["maximum"] <= 1
            ):
                return "INVALID_OBJECTIVE"
        elif operator == "LogMSE":
            if loss_transformation != "Sigmoid":
                return "INVALID_OBJECTIVE"
            if not (
                constraint["kind"] == "ClosedInterval"
                and constraint["minimum"] >= 0
            ):
                return "INVALID_OBJECTIVE"
    if len(direct) != len(slots):
        return "INVALID_OBJECTIVE"

    used_resources: set[str] = set()
    gradient_resources: set[str] = set()
    for component in auxiliary:
        operator = component["operator"]
        if operator not in known_auxiliary:
            return "UNKNOWN_PRIMITIVE"
        if available_auxiliary is not None and operator not in available_auxiliary:
            return "PRIMITIVE_UNAVAILABLE"
        roles = component["roles"]
        if operator == "GaussianNLL":
            if (
                roles["locationEstimate"]["identity"]
                != roles["observedLocation"]["identity"]
            ):
                return "INVALID_OBJECTIVE"
            if roles["locationEstimate"]["identity"] not in slot_by_id:
                return "INVALID_OBJECTIVE"
            identity = roles["scale"]["identity"]
            used_resources.add(identity)
            gradient_resources.add(identity)
        elif operator == "RiskAdjustedExpectedValue":
            used_resources.add(roles["uncertaintyScale"]["identity"])
        if operator in {"ExpectedValue", "RiskAdjustedExpectedValue"}:
            positive = roles["positiveOutcomeProbability"]["identity"]
            negative = roles["negativeOutcomeProbability"]["identity"]
            if positive == negative:
                return "INVALID_OBJECTIVE"
            if positive not in slot_by_id or negative not in slot_by_id:
                return "INVALID_OBJECTIVE"
            if any(
                slot_by_id[identity]["publicPredictionTransformation"]["kind"]
                != "Sigmoid"
                for identity in (positive, negative)
            ):
                return "INVALID_OBJECTIVE"

    if used_resources != set(resource_ids):
        return "INVALID_RESOURCE_GRAPH"
    if not set(resource_ids).issubset(gradient_resources):
        return "INVALID_RESOURCE_GRAPH"

    config = model_contract["modelConfig"]
    if config["hidden"] % config["nhead"]:
        return "INVALID_MODEL_CONTRACT"

    return None


def test_all_new_schemas_are_valid_and_all_references_resolve(
    schema_registry: Registry,
):
    paths = _schema_paths()
    assert len(paths) >= 30

    for path in paths:
        schema = _read(path)
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False

        resolver = schema_registry.resolver(base_uri=schema["$id"])
        pending = [schema]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str):
                    resolver.lookup(reference)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)


def test_semantic_fixtures_are_closed_valid_and_have_exact_d1_digests(
    schema_registry: Registry,
):
    fixture_schema = SEMANTIC_ROOT / "schemas" / "fixture.schema.json"
    model_schema = SEMANTIC_ROOT / "schemas" / "model-contract.schema.json"
    for name in SEMANTIC_FIXTURES:
        fixture = _fixture(name)
        _validate(fixture, fixture_schema, schema_registry)
        _validate(fixture["modelContract"], model_schema, schema_registry)
        assert _semantic_reason(fixture["modelContract"]) is None
        assert _d1(fixture["modelContract"]) == {
            key: fixture["expected"][key]
            for key in (
                "targetContractSha256",
                "objectiveSha256",
                "modelContractSha256",
            )
        }


@pytest.mark.parametrize("fixture_name", SEMANTIC_FIXTURES)
def test_d1_canonical_bytes_and_digests_are_identical_in_python_and_node(
    fixture_name: str,
):
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the contract test"
    fixture_path = SEMANTIC_ROOT / "fixtures" / fixture_name
    result = subprocess.run(
        [node, str(SEMANTIC_ROOT / "fixtures" / "d1_sha256.mjs"), str(fixture_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    model_contract = _read(fixture_path)["modelContract"]
    actual = json.loads(result.stdout)
    assert {
        key: actual[key]
        for key in (
            "targetContractSha256",
            "objectiveSha256",
            "modelContractSha256",
        )
    } == _d1(model_contract)
    assert actual["canonicalJson"] == {
        key: rfc8785.dumps(value).decode("utf-8")
        for key, value in _d1_preimages(model_contract).items()
    }


def test_offline_fixture_manifest_has_exact_sorted_file_digests(
    schema_registry: Registry,
):
    fixture_root = SEMANTIC_ROOT / "fixtures"
    manifest = _read(fixture_root / "manifest.json")
    _validate(
        manifest,
        SEMANTIC_ROOT / "schemas" / "fixture-manifest.schema.json",
        schema_registry,
    )
    _validate(
        _fixture("negative-cases.json"),
        SEMANTIC_ROOT / "schemas" / "negative-cases.schema.json",
        schema_registry,
    )
    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == sorted(paths)
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((fixture_root / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    manifest_digest, manifest_name = (fixture_root / "manifest.sha256").read_text(
        encoding="ascii"
    ).split()
    assert manifest_name == "manifest.json"
    assert manifest_digest == hashlib.sha256(
        (fixture_root / manifest_name).read_bytes()
    ).hexdigest()

    duplicate_fixture = fixture_root / "invalid" / "duplicate-object-key.json"

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads(
            duplicate_fixture.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )


def test_flight_fixture_manifest_has_exact_sorted_file_digests(
    schema_registry: Registry,
):
    fixture_root = FLIGHT_ROOT / "fixtures"
    manifest = _read(fixture_root / "manifest.json")
    _validate(
        manifest,
        FLIGHT_ROOT / "schemas" / "fixture-manifest.schema.json",
        schema_registry,
    )
    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == sorted(paths)
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((fixture_root / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    manifest_digest, manifest_name = (fixture_root / "manifest.sha256").read_text(
        encoding="ascii"
    ).split()
    assert manifest_name == "manifest.json"
    assert manifest_digest == hashlib.sha256(
        (fixture_root / manifest_name).read_bytes()
    ).hexdigest()


def test_semantic_negative_and_evolution_cases_are_classified(
    schema_registry: Registry,
):
    model_schema = SEMANTIC_ROOT / "schemas" / "model-contract.schema.json"

    bounded_identity = deepcopy(_fixture("single-probability.json")["modelContract"])
    bounded_identity["targetContract"]["slots"][0]["publicPredictionTransformation"] = {
        "kind": "Identity"
    }
    assert _semantic_reason(bounded_identity) == "INVALID_TARGET_CONTRACT"

    stop_gradient_only = deepcopy(
        _fixture("multi-target-shared-resource.json")["modelContract"]
    )
    del stop_gradient_only["objective"]["auxiliaryComponents"][0]
    del stop_gradient_only["objective"]["auxiliaryComponents"][0]
    assert _semantic_reason(stop_gradient_only) == "INVALID_RESOURCE_GRAPH"

    distinct_resources = deepcopy(
        _fixture("multi-target-shared-resource.json")["modelContract"]
    )
    distinct_resources["objective"]["resources"] = [
        {"identity": "scaleA", "kind": "PositiveScalarPerObservation"},
        {"identity": "scaleB", "kind": "PositiveScalarPerObservation"},
    ]
    auxiliaries = distinct_resources["objective"]["auxiliaryComponents"]
    auxiliaries[1]["roles"]["scale"]["identity"] = "scaleA"
    auxiliaries[2]["roles"]["uncertaintyScale"]["identity"] = "scaleA"
    second_gaussian = deepcopy(auxiliaries[1])
    second_gaussian["identity"] = "aux.gaussian-nll-b"
    second_gaussian["roles"]["scale"]["identity"] = "scaleB"
    auxiliaries.insert(2, second_gaussian)
    assert _semantic_reason(distinct_resources) is None

    probability = _fixture("single-probability.json")["modelContract"]
    assert (
        _semantic_reason(probability, available_direct={"SmoothL1"})
        == "PRIMITIVE_UNAVAILABLE"
    )

    unknown = deepcopy(probability)
    unknown["objective"]["directComponents"][0] = {
        "identity": "direct.event-probability",
        "operator": "ConsumerAddedLoss",
        "weight": 1,
        "roles": {
            "estimate": {
                "kind": "target",
                "identity": "EventProbability",
                "view": "lossEstimate",
            },
            "observed": {
                "kind": "target",
                "identity": "EventProbability",
                "view": "observed",
            },
        },
        "parameters": {},
    }
    _validate(unknown, model_schema, schema_registry)
    assert _semantic_reason(unknown) == "UNKNOWN_PRIMITIVE"

    reorder_a = _fixture("target-reorder-a-b.json")["expected"]
    reorder_b = _fixture("target-reorder-b-a.json")["expected"]
    assert reorder_a["targetContractSha256"] != reorder_b["targetContractSha256"]
    assert reorder_a["modelContractSha256"] != reorder_b["modelContractSha256"]

    changed_transformation = deepcopy(probability)
    changed_transformation["targetContract"]["slots"][0][
        "lossInputTransformation"
    ] = {"kind": "Sigmoid"}
    assert (
        _d1(changed_transformation)["targetContractSha256"]
        != _d1(probability)["targetContractSha256"]
    )

    changed_binding = deepcopy(distinct_resources)
    changed_binding["objective"]["auxiliaryComponents"][3]["roles"][
        "uncertaintyScale"
    ]["identity"] = "scaleB"
    assert (
        _d1(changed_binding)["objectiveSha256"]
        != _d1(distinct_resources)["objectiveSha256"]
    )


def test_numerical_golden_cases_match_operator_semantics():
    regression = _fixture("single-regression.json")["numericalCases"][0]
    raw = regression["inputs"]["rawCoordinate"]
    observed = regression["inputs"]["observed"]
    estimate = math.tanh(raw)
    loss = 0.5 * (estimate - observed) ** 2
    assert estimate == pytest.approx(
        regression["outputs"]["publicPrediction"], abs=regression["absoluteTolerance"]
    )
    assert loss == pytest.approx(
        regression["outputs"]["componentLoss"], abs=regression["absoluteTolerance"]
    )

    probability = _fixture("single-probability.json")["numericalCases"][0]
    raw = probability["inputs"]["rawCoordinate"]
    prediction = 1 / (1 + math.exp(-raw))
    bce = (
        max(raw, 0)
        - raw * probability["inputs"]["observed"]
        + math.log1p(math.exp(-abs(raw)))
    )
    assert prediction == probability["outputs"]["publicPrediction"]
    assert bce == pytest.approx(probability["outputs"]["componentLoss"])

    positive = _fixture("single-positive-regression.json")["numericalCases"][0]
    raw = positive["inputs"]["rawCoordinate"]
    observed = positive["inputs"]["observed"]
    estimate = 1 / (1 + math.exp(-raw))
    log_mse = (math.log(estimate + 1e-6) - math.log(observed + 1e-6)) ** 2
    assert estimate == positive["outputs"]["publicPrediction"]
    assert log_mse == pytest.approx(
        positive["outputs"]["componentLoss"], abs=positive["absoluteTolerance"]
    )

    shared = _fixture("multi-target-shared-resource.json")["numericalCases"][0]
    values = shared["inputs"]
    delta = values["positiveOutcomeProbability"] - values["negativeOutcomeProbability"]
    expected_value = -delta
    variance = values["sharedScale"] ** 2 + 1e-6
    gaussian_nll = 0.5 * (
        (values["observedLocation"] - values["locationEstimate"]) ** 2 / variance
        + math.log(variance)
    )
    risk = -(delta - 0.1 * values["sharedScale"] * abs(delta))
    assert expected_value == pytest.approx(shared["outputs"]["expectedValueLoss"])
    assert gaussian_nll == pytest.approx(
        shared["outputs"]["gaussianNllLoss"], abs=shared["absoluteTolerance"]
    )
    assert risk == pytest.approx(shared["outputs"]["riskAdjustedExpectedValueLoss"])


def test_hybrid_envelope_accepts_unknown_revision_before_registry_selection(
    schema_registry: Registry,
):
    model = deepcopy(_fixture("single-regression.json")["modelContract"])
    model["objectiveLanguageRevision"] = 2
    _validate(
        model,
        SEMANTIC_ROOT / "schemas" / "model-contract-envelope.schema.json",
        schema_registry,
    )
    with pytest.raises(ValidationError):
        _validate(
            model,
            SEMANTIC_ROOT / "schemas" / "model-contract.schema.json",
            schema_registry,
        )


def test_flight_capabilities_and_structured_errors_are_normative(
    schema_registry: Registry,
):
    capability = _read(FLIGHT_ROOT / "fixtures" / "json" / "capabilities.result.json")
    _validate(
        capability,
        FLIGHT_ROOT / "schemas" / "capabilities-result.schema.json",
        schema_registry,
    )
    language = capability["semantic"]["objectiveLanguage"]
    for key in (
        "constraints",
        "transformations",
        "resourceKinds",
        "directOperators",
        "auxiliaryOperators",
        "aggregations",
        "reductions",
    ):
        assert language[key] == sorted(language[key])
    encoded = json.dumps(capability)
    assert "MeanReturn" not in encoded
    assert "inventory." not in encoded
    assert capability["semantic"]["semanticLimits"]["maxTargetSlots"] != 6

    error_fixtures = sorted(
        (FLIGHT_ROOT / "fixtures" / "json").glob("error.*.json")
    )
    assert len(error_fixtures) == 8
    for path in error_fixtures:
        _validate(
            _read(path),
            FLIGHT_ROOT / "schemas" / "error-detail.schema.json",
            schema_registry,
        )

    worker_capability = {
        "contract": "transformer-worker",
        "protocolVersion": 12,
        "checkpointFormat": "transformer-checkpoint-v6",
        "recoveryFormat": "transformer-recovery-v6",
        "schemaIds": capability["schemaIds"],
        "semantic": capability["semantic"],
        "torchVersion": "2.12.0",
        "cudaRuntimeVersion": "13.0",
        "devices": [
            {"kind": "cpu", "opaqueId": "cpu", "name": "CPU"},
            {"kind": "cuda", "opaqueId": "cuda:0", "name": "GPU 0"},
        ],
    }
    _validate(
        worker_capability,
        WORKER_ROOT / "schemas" / "capabilities.schema.json",
        schema_registry,
    )


@pytest.mark.parametrize("fixture_name", ("fit", "predict", "published-model"))
def test_job_config_preimage_has_exact_cross_language_jcs_digest(
    fixture_name: str,
    schema_registry: Registry,
):
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the contract test"
    path = FLIGHT_ROOT / "fixtures" / "json" / f"job-config.{fixture_name}.json"
    fixture = _read(path)
    _validate(
        fixture,
        FLIGHT_ROOT / "schemas" / "job-config-fixture.schema.json",
        schema_registry,
    )
    result = subprocess.run(
        [
            node,
            str(SEMANTIC_ROOT / "fixtures" / "jcs_sha256.mjs"),
            str(path),
            "jobConfig",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)
    assert actual["canonicalJson"] == rfc8785.dumps(fixture["jobConfig"]).decode(
        "utf-8"
    )
    assert actual["sha256"] == fixture["expectedJobConfigSha256"]
    initialization = fixture["jobConfig"].get("initialization")
    if initialization is not None and initialization["kind"] == "publishedModel":
        for layer in ("DataContract", "TargetContract", "Objective", "ModelContract"):
            current = f"{layer[0].lower()}{layer[1:]}Sha256"
            assert initialization[f"parent{layer}Sha256"] == initialization[current]


def _reconstruct(fixture: dict) -> list[list[list[float]]]:
    encoding = fixture["sourceEncoding"]
    position = 0
    for definition in encoding["featureBlocks"]:
        if definition["position"] != position:
            raise ValueError("INVALID_SOURCE_ENCODING")
        position += definition["windowRows"] * definition["nativeRowWidth"]
    if position != fixture["featureDim"]:
        raise ValueError("INVALID_SOURCE_ENCODING")

    reconstructed = []
    previous_range = None
    next_example_offset = None
    for chunk in fixture["chunks"]:
        if len(chunk["blocks"]) != len(encoding["featureBlocks"]):
            raise ValueError("INVALID_SOURCE_ENCODING")

        logical_rows = len(chunk["blocks"][0]["observationOffsets"])
        if logical_rows == 0 or any(
            len(block["observationOffsets"]) != logical_rows
            for block in chunk["blocks"]
        ):
            raise ValueError("INVALID_SOURCE_ENCODING")

        range_ordinal = chunk["rangeOrdinal"]
        example_offset = chunk["exampleOffset"]
        if previous_range is None:
            if range_ordinal != 0 or example_offset != 0:
                raise ValueError("INVALID_SOURCE_ENCODING")
        elif range_ordinal == previous_range:
            if example_offset != next_example_offset:
                raise ValueError("INVALID_SOURCE_ENCODING")
        elif range_ordinal != previous_range + 1 or example_offset != 0:
            raise ValueError("INVALID_SOURCE_ENCODING")

        for example in range(logical_rows):
            sequence = []
            for sequence_index in range(fixture["seqLen"]):
                observation = []
                for definition, block in zip(
                    encoding["featureBlocks"], chunk["blocks"], strict=True
                ):
                    offsets = block["observationOffsets"][example]
                    if len(offsets) != fixture["seqLen"]:
                        raise ValueError("INVALID_SOURCE_ENCODING_OFFSET")
                    start = block["observationOffsets"][example][sequence_index]
                    stop = start + definition["windowRows"]
                    if stop > len(block["nativeRows"]):
                        raise ValueError("INVALID_SOURCE_ENCODING_OFFSET")
                    for native_row in block["nativeRows"][start:stop]:
                        if len(native_row) != definition["nativeRowWidth"] or any(
                            not math.isfinite(value) for value in native_row
                        ):
                            raise ValueError("INVALID_SOURCE_ENCODING_OFFSET")
                        observation.extend(native_row)
                sequence.append(observation)
            reconstructed.append(sequence)

        previous_range = range_ordinal
        next_example_offset = example_offset + logical_rows

    return reconstructed


def test_indexed_feature_blocks_remains_provider_neutral_and_exact(
    schema_registry: Registry,
):
    schema = FLIGHT_ROOT / "schemas" / "indexed-feature-blocks-fixture.schema.json"
    root = FLIGHT_ROOT / "fixtures" / "indexed-feature-blocks"
    accepted_names = (
        "single-block.json",
        "heterogeneous-multi-block.json",
        "non-contiguous-native-offsets.json",
        "payload-boundary-one-chunk.json",
        "payload-boundary-two-chunks.json",
    )
    for name in accepted_names:
        fixture = _read(root / name)
        _validate(fixture, schema, schema_registry)
        assert _reconstruct(fixture) == fixture["expected"]["logicalFeatures"]

    one_chunk = _read(root / "payload-boundary-one-chunk.json")
    two_chunks = _read(root / "payload-boundary-two-chunks.json")
    assert _reconstruct(one_chunk) == _reconstruct(two_chunks)
    assert sum(
        len(block["nativeRows"])
        for chunk in one_chunk["chunks"]
        for block in chunk["blocks"]
    ) == 4
    assert sum(
        len(block["nativeRows"])
        for chunk in two_chunks["chunks"]
        for block in chunk["blocks"]
    ) == 6

    missing = _read(root / "missing-native-prefix.json")
    _validate(missing, schema, schema_registry)
    with pytest.raises(ValueError, match="INVALID_SOURCE_ENCODING_OFFSET"):
        _reconstruct(missing)

    contents = "\n".join(path.read_text() for path in root.glob("*.json"))
    assert "production-core" not in contents
    assert "inventory" not in contents.lower()


def test_checkpoint_recovery_and_metrics_fixtures_match_new_schemas(
    schema_registry: Registry,
):
    _validate(
        _read(CHECKPOINT_ROOT / "fixtures" / "checkpoint-artifact.json"),
        CHECKPOINT_ROOT / "schemas" / "checkpoint-artifact.schema.json",
        schema_registry,
    )
    _validate(
        _read(CHECKPOINT_ROOT / "fixtures" / "recovery-metadata.json"),
        CHECKPOINT_ROOT / "schemas" / "recovery-metadata.schema.json",
        schema_registry,
    )
    training_record = _read(METRICS_ROOT / "fixtures" / "training-record.json")
    _validate(
        training_record,
        METRICS_ROOT / "training-record.schema.json",
        schema_registry,
    )
    assert [item["index"] for item in training_record["targets"]] == list(
        range(len(training_record["targets"]))
    )
    assert [
        (item["targetIndex"], item["targetIdentity"])
        for item in training_record["directLosses"]
    ] == [
        (item["index"], item["identity"]) for item in training_record["targets"]
    ]
    assert [
        (item["targetIndex"], item["targetIdentity"])
        for item in training_record["targetMetrics"]
    ] == [
        (item["index"], item["identity"]) for item in training_record["targets"]
    ]

    worker_metrics_schema = _read(
        WORKER_ROOT / "schemas" / "training-metrics.schema.json"
    )
    worker_metrics = {
        key: training_record[key] for key in worker_metrics_schema["required"]
    }
    _validate(
        worker_metrics,
        WORKER_ROOT / "schemas" / "training-metrics.schema.json",
        schema_registry,
    )

    run_summary = _read(FIT_RUN_ROOT / "fixtures" / "run-summary.json")
    _validate(
        run_summary,
        FIT_RUN_ROOT / "run-summary.schema.json",
        schema_registry,
    )
    assert [item["index"] for item in run_summary["targets"]] == list(
        range(len(run_summary["targets"]))
    )

    point = {
        "schema": "transformer.metrics-point.v5",
        "eventId": SHA_A,
        "documentSha256": SHA_B,
        "recordedAt": training_record["recordedAt"],
        "deploymentId": "test-deployment",
        "runId": training_record["jobId"],
        "attemptId": training_record["attemptId"],
        "attempt": training_record["attempt"],
        "modelRef": None,
        "semanticDigests": training_record["semanticDigests"],
        "epoch": training_record["epoch"],
        "step": training_record["step"],
        "target": {"identity": "ConsumerDefined.EventProbability", "index": 0},
        "component": {
            "identity": "direct.consumer-event",
            "operator": "BinaryCrossEntropyWithLogits",
        },
        "metric": {"name": "training.loss.direct", "value": 0.5, "unit": "1"},
    }
    _validate(point, METRICS_ROOT / "point.schema.json", schema_registry)
    invalid_point = {
        **point,
        "pair": {
            "leftComponentIdentity": "direct.a",
            "rightComponentIdentity": "direct.b",
        },
    }
    with pytest.raises(ValidationError):
        _validate(invalid_point, METRICS_ROOT / "point.schema.json", schema_registry)

    run_document = {
        "schema": "transformer.metrics-fit-run.v5",
        "summaryId": SHA_A,
        "deploymentId": "test-deployment",
        "runId": run_summary["jobId"],
        "modelRef": run_summary["modelRef"],
        "recordedAt": run_summary["recordedAt"],
        "semanticDigests": run_summary["semanticDigests"],
        "jobConfigSha256": run_summary["jobConfigSha256"],
        "checkpointFormat": run_summary["checkpointFormat"],
        "targets": run_summary["targets"],
        "initialization": run_summary["initialization"],
        "milestones": run_summary["milestones"],
        "durations": run_summary["durations"],
        "counts": run_summary["counts"],
        "artifact": {"byteCount": 1024, "sha256": SHA_B},
    }
    _validate(
        run_document,
        FIT_RUN_ROOT / "run-document.schema.json",
        schema_registry,
    )

    for path, pattern in (
        (
            METRICS_ROOT / "opensearch" / "metrics-points-v5.template.json",
            "metrics-points-v5",
        ),
        (
            FIT_RUN_ROOT / "opensearch" / "metrics-runs-v5.template.json",
            "metrics-runs-v5",
        ),
    ):
        template = _read(path)
        assert template["index_patterns"] == [pattern]
        assert template["template"]["settings"] == {"number_of_replicas": 0}
        assert template["template"]["mappings"]["dynamic"] == "strict"


def test_flight_worker_and_checkpoint_envelopes_compose(
    schema_registry: Registry,
):
    fixture = _fixture("single-regression.json")
    model_contract = fixture["modelContract"]
    semantic_digests = {"dataContractSha256": SHA_A, **_d1(model_contract)}
    data_contract = {
        "identity": "consumer.learning-dataset",
        "revision": 1,
        "profile": "consumer.profile.alpha",
        "dataContractSha256": SHA_A,
        "seqLen": 10,
        "featureDim": 64848,
    }
    source_encoding = {
        "kind": "indexedFeatureBlocks",
        "featureBlocks": [
            {"position": 0, "windowRows": 1, "nativeRowWidth": 64848},
        ],
    }
    training = {
        "lr": 0.001,
        "batchSize": 64,
        "epochs": 2,
        "useAmp": True,
        "weightDecay": 0,
        "selection": None,
        "seed": 42,
        "deterministic": True,
    }
    diagnostics = {"schemaVersion": 1, "gradientInteractions": None}

    request = {
        "contract": "transformer-flight",
        "version": 11,
        "requestId": "11111111-1111-4111-8111-111111111111",
        "idempotencyKey": "fit-1",
        "jobId": JOB_ID,
        "clientExecutionId": ATTEMPT_ID,
        "operation": "fit",
        "device": "auto",
        "modelLabel": "alpha",
        "initialization": {"kind": "random"},
        "sourceEncoding": source_encoding,
        "dataContract": data_contract,
        "modelContract": model_contract,
    }
    _validate(request, FLIGHT_ROOT / "schemas" / "create.schema.json", schema_registry)

    command = {
        "contract": "transformer-worker",
        "protocolVersion": 12,
        "jobId": JOB_ID,
        "attempt": 1,
        "attemptId": ATTEMPT_ID,
        "operation": "fit",
        "device": {"kind": "cuda", "opaqueId": "cuda:0"},
        "inputs": [],
        "inputRevision": 0,
        "inputClosed": False,
        "manifestSha256": None,
        "workspace": {"root": "/runtime/jobs/example"},
        "model": {"label": "alpha"},
        "sourceEncoding": source_encoding,
        "dataContract": data_contract,
        "modelContract": model_contract,
        "semanticDigests": semantic_digests,
        "jobConfigSha256": SHA_B,
        "training": training,
        "diagnostics": diagnostics,
        "initialization": {"kind": "random"},
        "recovery": None,
    }
    _validate(
        command,
        WORKER_ROOT / "schemas" / "command-manifest.schema.json",
        schema_registry,
    )

    predict_command = {
        **command,
        "operation": "predict",
        "predictionColumn": "prediction",
        "model": {
            "checkpoint": {
                "path": "/runtime/models/example/checkpoint.pth",
                "format": "transformer-checkpoint-v6",
                "byteCount": 1024,
                "checkpointSha256": SHA_C,
            }
        },
    }
    for key in ("training", "diagnostics", "initialization", "recovery"):
        del predict_command[key]

    _validate(
        predict_command,
        WORKER_ROOT / "schemas" / "command-manifest.schema.json",
        schema_registry,
    )

    empty_checkpoint = deepcopy(predict_command)
    empty_checkpoint["model"]["checkpoint"]["byteCount"] = 0
    with pytest.raises(ValidationError):
        _validate(
            empty_checkpoint,
            WORKER_ROOT / "schemas" / "command-manifest.schema.json",
            schema_registry,
        )

    checkpoint_metadata = {
        "format": "transformer-checkpoint-v6",
        "serviceVersion": "0.2.0",
        "generation": 1,
        "jobId": JOB_ID,
        "dataContract": data_contract,
        "modelContract": model_contract,
        "semanticDigests": semantic_digests,
        "trainingConfig": training,
        "diagnostics": diagnostics,
        "selection": {
            "enabled": False,
            "modelContractSha256": semantic_digests["modelContractSha256"],
            "bestSelectionScore": None,
            "bestEpoch": None,
            "source": "last_epoch",
        },
        "initialization": {"kind": "random"},
        "jobConfigSha256": SHA_B,
        "manifestSha256": SHA_C,
        "progress": {
            "completedEpochs": 2,
            "globalStep": 200,
            "trainingComplete": True,
        },
    }
    _validate(
        checkpoint_metadata,
        CHECKPOINT_ROOT / "schemas" / "checkpoint-metadata.schema.json",
        schema_registry,
    )

    serialized = json.dumps({request["dataContract"]["identity"]: model_contract})
    assert "MeanReturn" in serialized  # opaque fixture value remains round-trippable
    assert "inventory." not in serialized
