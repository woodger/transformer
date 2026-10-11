import json
from copy import deepcopy

import pytest

from app.contracts.flight.v24.constants import FIT_CREATE_ACTION
from app.contracts.semantic.v7 import ModelContract, SemanticContractError
from app.contracts.semantic.v7.capabilities import semantic_capabilities
from app.contracts.semantic.v7.schema import validate_schema
from app.contracts.worker.v22.model_config import ModelConfig
from app.contracts.worker.v22.model_definition import resolved_semantic_digests
from app.project import PROJECT_ROOT
from app.service.adapters.inbound.flight.validation import validate_action_request
from tests.fixture_documents import semantic_fixture_document


def _document():
    return semantic_fixture_document("weighted-binary-confidence-penalty")["modelContract"]


@pytest.mark.parametrize(("weight", "reason"), [
    (0, "INVALID_OBJECTIVE"),
    (-0.1, "INVALID_OBJECTIVE"),
    (True, "INVALID_OBJECTIVE"),
    (float("inf"), "INVALID_MODEL_CONTRACT"),
    (float("nan"), "INVALID_MODEL_CONTRACT"),
])
def test_confidence_penalty_requires_a_positive_finite_coefficient(weight, reason):
    document = _document()
    document["objective"]["auxiliaryComponents"][0]["weight"] = weight

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == reason


def test_confidence_penalty_rejects_an_unknown_probability_target():
    document = _document()
    document["objective"]["auxiliaryComponents"][0]["roles"]["probability"] = "unknown"

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"
    assert error.value.path == "/objective/auxiliaryComponents/0/roles"


def test_confidence_penalty_rejects_a_non_probability_output():
    document = semantic_fixture_document("single-regression")["modelContract"]
    target = document["targetContract"]["slots"][0]["identity"]
    document["objective"]["auxiliaryComponents"] = [{
        "identity": "confidence",
        "operator": "BernoulliConfidencePenalty",
        "weight": 0.1,
        "roles": {"probability": target},
    }]

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"
    assert error.value.path == "/objective/auxiliaryComponents/0/roles"


def test_confidence_penalty_and_its_weight_are_bound_to_model_identity():
    document = _document()
    config = ModelConfig.from_tuning(document["modelTuning"], seq_len=2, feature_dim=8)
    definitions = []
    for coefficient in (None, 0.1, 0.2):
        candidate = deepcopy(document)
        if coefficient is None:
            candidate["objective"].pop("auxiliaryComponents")
        else:
            candidate["objective"]["auxiliaryComponents"][0]["weight"] = coefficient
        contract = ModelContract.from_document(candidate)
        definitions.append(resolved_semantic_digests(contract, "a" * 64, config))

    assert len({item["objectiveSha256"] for item in definitions}) == 3
    assert len({item["modelDefinitionSha256"] for item in definitions}) == 3
    assert len({item["targetContractSha256"] for item in definitions}) == 1


def test_semantic_capabilities_advertise_the_confidence_penalty():
    capabilities = semantic_capabilities()

    validate_schema(capabilities, "language-capabilities")

    assert capabilities["objectiveLanguage"]["revision"] == 7
    assert "BernoulliConfidencePenalty" in capabilities["auxiliaryOperators"]


def test_flight_accepts_a_weighted_binary_confidence_penalty():
    path = PROJECT_ROOT / "app/contracts/flight/v24/fixtures/json/fit-create.confidence-penalty.json"
    document = json.loads(path.read_text())

    fields = validate_action_request(FIT_CREATE_ACTION, document)

    contract = ModelContract.from_document(fields["model_contract"])
    assert contract.auxiliary_components[0]["operator"] == (
        "BernoulliConfidencePenalty"
    )
