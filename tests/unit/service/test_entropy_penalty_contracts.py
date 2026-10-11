import json
import subprocess
from copy import deepcopy

import pytest

from app.contracts.checkpoint.v14.codec import validate_checkpoint_document
from app.contracts.flight.v24.constants import FIT_CREATE_ACTION
from app.contracts.model_catalog.v9.codec import validate_catalog_document
from app.contracts.semantic.v6 import (
    ModelContract as PreviousModelContract,
    SemanticContractError as PreviousContractError,
)
from app.contracts.semantic.v7 import (
    ModelContract,
    SemanticContractError,
    semantic_capabilities,
)
from app.contracts.semantic.v7.schema import validate_schema
from app.contracts.training_telemetry.v4.codec import (
    validate_training_telemetry_document,
)
from app.contracts.worker.v22.codec import validate_document
from app.contracts.worker.v22.model_config import ModelConfig
from app.contracts.worker.v22.model_definition import resolved_semantic_digests
from app.project import PROJECT_ROOT
from app.service.adapters.inbound.flight.validation import validate_action_request
from tests.fixture_documents import semantic_fixture_document
from tests.support.internal_contract_documents import internal_contract_documents


def _document():
    return semantic_fixture_document("weighted-binary-entropy-penalty")["modelContract"]


@pytest.mark.parametrize(
    "weight", [0, -0.1, True, float("inf"), float("-inf"), float("nan")]
)
def test_entropy_penalty_rejects_nonpositive_or_nonfinite_weights(weight):
    document = _document()
    document["objective"]["auxiliaryComponents"][0]["weight"] = weight

    with pytest.raises(SemanticContractError):
        ModelContract.from_document(document)


@pytest.mark.parametrize("transformation", ["Identity", "Tanh"])
def test_entropy_penalty_requires_a_public_probability(transformation):
    document = semantic_fixture_document("single-regression")["modelContract"]
    slot = document["targetContract"]["slots"][0]
    slot["publicPredictionTransformation"] = transformation
    if transformation == "Identity":
        slot.pop("observedConstraint")
    ModelContract.from_document(document)

    document["objective"]["auxiliaryComponents"] = [
        {
            "identity": "auxiliary.entropy",
            "operator": "BernoulliEntropyPenalty",
            "weight": 0.1,
            "roles": {"probability": slot["identity"]},
        }
    ]

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"
    assert error.value.path == "/objective/auxiliaryComponents/0/roles"


def test_entropy_penalty_rejects_unknown_probability_bindings():
    document = _document()
    document["objective"]["auxiliaryComponents"][0]["roles"]["probability"] = (
        "unknown.target"
    )

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parameters", {}),
        ("lambda", 0.1),
        ("sign", -1),
    ],
)
def test_entropy_penalty_rejects_undeclared_configuration_fields(field, value):
    document = _document()
    document["objective"]["auxiliaryComponents"][0][field] = value

    with pytest.raises(SemanticContractError):
        ModelContract.from_document(document)


def test_entropy_penalty_rejects_an_observation_role():
    document = _document()
    component = document["objective"]["auxiliaryComponents"][0]
    component["roles"]["observed"] = component["roles"]["probability"]

    with pytest.raises(SemanticContractError):
        ModelContract.from_document(document)


def test_entropy_penalty_requires_its_probability_role():
    document = _document()
    document["objective"]["auxiliaryComponents"][0]["roles"] = {}

    with pytest.raises(SemanticContractError):
        ModelContract.from_document(document)


@pytest.mark.parametrize(
    "operator", ["GaussianNLL", "ExpectedValue", "RiskAdjustedExpectedValue"]
)
def test_weighted_binary_targets_keep_their_other_auxiliary_role_restrictions(operator):
    document = semantic_fixture_document("multi-target-shared-resource")[
        "modelContract"
    ]
    direct = document["objective"]["directComponents"][1]
    direct["operator"] = "PositiveClassWeightedBinaryCrossEntropyWithLogits"
    direct["positiveClassWeight"] = 28
    component = next(
        item
        for item in document["objective"]["auxiliaryComponents"]
        if item["operator"] == operator
    )
    if operator == "GaussianNLL":
        component["roles"]["locationEstimate"] = direct["targetIdentity"]
        component["roles"]["observedLocation"] = direct["targetIdentity"]
    document["objective"]["auxiliaryComponents"] = [component]

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"
    assert "weighted binary targets" in str(error.value)


def test_entropy_penalty_is_advertised_by_semantic_v7_and_rejected_by_v6():
    capabilities = semantic_capabilities()

    validate_schema(capabilities, "language-capabilities")

    assert capabilities["objectiveLanguage"] == {"revision": 7, "closed": True}
    assert "BernoulliConfidencePenalty" in capabilities["auxiliaryOperators"]
    assert "BernoulliEntropyPenalty" in capabilities["auxiliaryOperators"]
    with pytest.raises(PreviousContractError):
        PreviousModelContract.from_document(_document())


def test_entropy_operator_and_weight_change_objective_identity_without_changing_predictions():
    document = _document()
    config = ModelConfig.from_tuning(document["modelTuning"], seq_len=2, feature_dim=8)
    contracts = []
    for operator, coefficient in (
        (None, None),
        ("BernoulliEntropyPenalty", 0.1),
        ("BernoulliEntropyPenalty", 0.2),
        ("BernoulliConfidencePenalty", 0.1),
    ):
        candidate = deepcopy(document)
        if operator is None:
            candidate["objective"].pop("auxiliaryComponents")
        else:
            component = candidate["objective"]["auxiliaryComponents"][0]
            component.update(operator=operator, weight=coefficient)
        contracts.append(ModelContract.from_document(candidate))

    definitions = [
        resolved_semantic_digests(contract, "a" * 64, config) for contract in contracts
    ]

    assert len({item["objectiveSha256"] for item in definitions}) == 4
    assert len({item["modelDefinitionSha256"] for item in definitions}) == 4
    assert len({item["targetContractSha256"] for item in definitions}) == 1
    assert all(
        contract.prediction_definition(2) == contracts[0].prediction_definition(2)
        for contract in contracts
    )


@pytest.mark.parametrize(
    "fixture", ["bernoulli-entropy-penalty", "weighted-binary-entropy-penalty"]
)
def test_entropy_penalty_d1_matches_the_javascript_conformance_implementation(fixture):
    path = PROJECT_ROOT / "app/contracts/semantic/v7/fixtures" / f"{fixture}.json"
    result = subprocess.run(
        ["node", str(path.parent / "d1_sha256.mjs"), str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    actual = json.loads(result.stdout)
    document = semantic_fixture_document(fixture)
    expected = document["expected"]

    assert actual["targetContractSha256"] == expected["targetContractSha256"]
    assert actual["objectiveSha256"] == expected["objectiveSha256"]


def test_entropy_penalty_survives_flight_worker_checkpoint_and_catalog_boundaries():
    path = (
        PROJECT_ROOT
        / "app/contracts/flight/v24/fixtures/json/fit-create.entropy-penalty.json"
    )
    fields = validate_action_request(FIT_CREATE_ACTION, json.loads(path.read_text()))
    contract = ModelContract.from_document(fields["model_contract"])
    documents = internal_contract_documents(
        checkpoint_version=14,
        worker_version=22,
        semantic_version=7,
        target_head_version=5,
    )
    command = documents["worker"]["command-manifest"]
    metadata = documents["checkpoint"]["checkpoint-metadata"]
    config = ModelConfig.from_manifest(command["modelConfig"])
    digests = resolved_semantic_digests(contract, "a" * 64, config)
    for document in (command, metadata):
        document["modelContract"] = contract.to_document()
        document["semanticDigests"] = digests
    metadata["selection"]["modelDefinitionSha256"] = digests["modelDefinitionSha256"]

    worker = validate_document(json.loads(json.dumps(command)), "command-manifest")
    checkpoint = validate_checkpoint_document(
        json.loads(json.dumps(metadata)), "checkpoint-metadata"
    )
    catalog = json.loads(
        (
            PROJECT_ROOT
            / "app/contracts/model_catalog/v9/fixtures/detail.result.entropy-penalty.json"
        ).read_text()
    )
    validate_catalog_document(catalog, "detail-result")

    assert worker["modelContract"] == contract.to_document()
    assert checkpoint["modelContract"] == contract.to_document()
    assert catalog["model"]["modelContract"] == contract.to_document()
    assert (
        worker["semanticDigests"]
        == checkpoint["semanticDigests"]
        == catalog["model"]["semanticDigests"]
    )


def test_public_telemetry_accepts_entropy_layout_and_positive_unweighted_losses():
    path = (
        PROJECT_ROOT
        / "app/contracts/training_telemetry/v4/fixtures/report.result.available.json"
    )
    report = json.loads(path.read_text())
    component = _document()["objective"]["auxiliaryComponents"][0]
    report["layout"]["auxiliaryComponents"] = [
        {"identity": component["identity"], "operator": component["operator"]}
    ]
    for epoch in (*report["anchors"], *report["epochPage"]["items"]):
        epoch["auxiliaryLosses"] = [0.6]
        epoch["totalLoss"] = 0.46

    validated = validate_training_telemetry_document(report, "report-result")

    assert validated["layout"]["auxiliaryComponents"][0]["operator"] == (
        "BernoulliEntropyPenalty"
    )
    assert validated["epochPage"]["items"][0]["auxiliaryLosses"] == [0.6]
