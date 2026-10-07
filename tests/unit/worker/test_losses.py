import math
from copy import deepcopy

import pytest
import torch

from app.contracts.semantic.v6 import ModelContract, SemanticContractError
from app.worker.model.transformer import public_predictions
from app.worker.training.losses import combined_loss
from tests.fixture_documents import semantic_fixture_document

MODEL_CONTRACT = ModelContract.from_document(
    semantic_fixture_document("multi-target-shared-resource")["modelContract"],
)


def make_outputs_and_targets():
    outputs = torch.tensor(
        [[0.1, -0.4, 0.7, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[-0.2, 0.8, 0.1]],
        dtype=torch.float64,
    )
    return outputs, targets


@pytest.mark.parametrize("target_index", range(3))
def test_declarative_objective_directly_supervises_every_selected_head(
    target_index,
):
    outputs, targets = make_outputs_and_targets()

    loss = combined_loss(outputs, targets, MODEL_CONTRACT)
    loss.backward()

    gradient = outputs.grad[0, target_index]
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0


@pytest.mark.parametrize(
    ("target_index", "replacement"),
    [(0, 0.6), (1, 0.2), (2, 0.8)],
)
def test_each_target_changes_only_its_corresponding_direct_component(
    target_index,
    replacement,
):
    outputs, targets = make_outputs_and_targets()
    changed_targets = targets.clone()
    changed_targets[0, target_index] = replacement

    original = combined_loss(
        outputs,
        targets,
        MODEL_CONTRACT,
        return_parts=True,
    ).statistics
    changed = combined_loss(
        outputs,
        changed_targets,
        MODEL_CONTRACT,
        return_parts=True,
    ).statistics

    assert changed.direct_losses[target_index] != pytest.approx(
        original.direct_losses[target_index]
    )
    for other_index in set(range(3)) - {target_index}:
        assert changed.direct_losses[other_index] == pytest.approx(
            original.direct_losses[other_index]
        )


def test_public_location_and_private_gaussian_scale_are_distinct_heads():
    outputs, targets = make_outputs_and_targets()

    loss = combined_loss(outputs, targets, MODEL_CONTRACT)
    loss.backward()

    assert outputs.grad[0, 0].abs() > 0
    assert outputs.grad[0, 3].abs() > 0


def test_probability_targets_use_logits_for_stable_direct_loss():
    outputs, targets = make_outputs_and_targets()

    statistics = combined_loss(
        outputs,
        targets,
        MODEL_CONTRACT,
        return_parts=True,
    ).statistics

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs[:, 1],
        targets[:, 1],
    )
    assert statistics.direct_losses[1] == pytest.approx(expected.item())


def test_positive_class_weighted_binary_bce_applies_weight_before_reduction():
    contract = _weighted_binary_contract(28)
    outputs = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float64)

    statistics = combined_loss(
        outputs,
        targets,
        contract,
        return_parts=True,
    ).statistics

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs[:, 0],
        targets[:, 0],
        pos_weight=torch.tensor(28.0, dtype=torch.float64),
    )
    assert statistics.direct_losses == pytest.approx((expected.item(),))


def test_positive_class_weighted_binary_bce_returns_corrected_probability():
    contract = _weighted_binary_contract(28)
    outputs = torch.tensor([[0.0], [2.0]], dtype=torch.float64)

    predictions = public_predictions(outputs, contract)

    assert predictions[:, 0] == pytest.approx(torch.sigmoid(
        outputs[:, 0] - torch.log(torch.tensor(28.0, dtype=torch.float64))
    ))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("positive_class_weight", [100000.0, 0.001])
def test_weighted_binary_loss_and_gradients_preserve_weight_with_reduced_precision(
    dtype,
    positive_class_weight,
):
    contract = _weighted_binary_contract(positive_class_weight)
    outputs = torch.zeros(2, 1, dtype=dtype, requires_grad=True)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss = combined_loss(outputs, targets, contract)
    loss.backward()

    assert loss.item() == pytest.approx(
        math.log(2) * (positive_class_weight + 1) / 2,
        rel=1e-6,
    )
    torch.testing.assert_close(
        outputs.grad.float(),
        torch.tensor([[-positive_class_weight / 4], [0.25]]),
        rtol=0.006,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("positive_class_weight", [100000.0, 1e-8])
def test_weighted_binary_probability_preserves_weight_with_reduced_precision(
    dtype,
    positive_class_weight,
):
    contract = _weighted_binary_contract(positive_class_weight)
    outputs = torch.tensor([[0.0], [2.0]], dtype=dtype)

    predictions = public_predictions(outputs, contract)

    torch.testing.assert_close(
        predictions,
        torch.tensor([
            [1 / (1 + positive_class_weight)],
            [math.exp(2) / (math.exp(2) + positive_class_weight)],
        ]),
        rtol=1e-6,
        atol=0,
    )


def test_positive_class_weighted_binary_bce_with_weight_one_matches_bce():
    weighted_contract = _weighted_binary_contract(1)
    unweighted_contract = ModelContract.from_document(
        semantic_fixture_document("single-probability")["modelContract"]
    )
    outputs = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float64)

    weighted = combined_loss(
        outputs,
        targets,
        weighted_contract,
        return_parts=True,
    ).statistics
    unweighted = combined_loss(
        outputs,
        targets,
        unweighted_contract,
        return_parts=True,
    ).statistics

    assert weighted.direct_losses == pytest.approx(unweighted.direct_losses)
    assert public_predictions(outputs, weighted_contract) == pytest.approx(
        public_predictions(outputs, unweighted_contract)
    )


@pytest.mark.parametrize(
    "fixture_id",
    [
        "positive-class-weighted-binary-w1",
        "positive-class-weighted-binary-w28",
    ],
)
def test_positive_class_weighted_binary_bce_matches_numerical_golden_cases(
    fixture_id,
):
    fixture = semantic_fixture_document(fixture_id)
    contract = ModelContract.from_document(fixture["modelContract"])
    cases = fixture["numericalCases"]
    assert isinstance(cases, list)

    for case in cases:
        assert isinstance(case, dict)
        inputs = case["inputs"]
        expected = case["outputs"]
        assert isinstance(inputs, dict)
        assert isinstance(expected, dict)
        raw_coordinate = float(inputs["rawCoordinate"])
        observed = float(inputs["observed"])
        tolerance = float(case["absoluteTolerance"])
        outputs = torch.tensor([[raw_coordinate]], dtype=torch.float64)
        targets = torch.tensor([[observed]], dtype=torch.float64)

        statistics = combined_loss(
            outputs,
            targets,
            contract,
            return_parts=True,
        ).statistics

        assert statistics.direct_losses[0] == pytest.approx(
            float(expected["componentLoss"]),
            abs=tolerance,
        )
        assert public_predictions(outputs, contract)[0, 0] == pytest.approx(
            float(expected["publicPrediction"]),
            abs=tolerance,
        )


def test_positive_class_weighted_binary_bce_requires_exact_binary_target_contract():
    document = _weighted_binary_contract(1).to_document()
    target_contract = document["targetContract"]
    assert isinstance(target_contract, dict)
    slots = target_contract["slots"]
    assert isinstance(slots, list)
    slots[0]["lossInputTransformation"] = "Tanh"

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"
    assert error.value.path == "/objective/directComponents/0"


@pytest.mark.parametrize("positive_class_weight", [0, -1, float("inf")])
def test_positive_class_weighted_binary_bce_requires_positive_finite_weight(
    positive_class_weight,
):
    document = _weighted_binary_contract(1).to_document()
    objective = document["objective"]
    assert isinstance(objective, dict)
    direct = objective["directComponents"]
    assert isinstance(direct, list)
    direct[0]["positiveClassWeight"] = positive_class_weight

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"


def test_positive_class_weighted_binary_target_cannot_participate_in_auxiliary_loss():
    document = semantic_fixture_document("multi-target-shared-resource")[
        "modelContract"
    ]
    assert isinstance(document, dict)
    objective = document["objective"]
    assert isinstance(objective, dict)
    direct = objective["directComponents"]
    assert isinstance(direct, list)
    direct[1]["operator"] = "PositiveClassWeightedBinaryCrossEntropyWithLogits"
    direct[1]["positiveClassWeight"] = 1

    with pytest.raises(SemanticContractError) as error:
        ModelContract.from_document(document)

    assert error.value.reason == "INVALID_OBJECTIVE"
    assert error.value.path == "/objective/auxiliaryComponents/0/roles"


def test_expected_value_and_risk_adjusted_expected_value_are_distinct():
    outputs, targets = make_outputs_and_targets()
    expected_value = _expected_value_only_contract()

    plain = combined_loss(
        outputs[:, :3],
        targets,
        expected_value,
        return_parts=True,
    ).statistics
    adjusted = combined_loss(
        outputs,
        targets,
        MODEL_CONTRACT,
        return_parts=True,
    ).statistics

    plain_value = next(
        value
        for _identity, operator, value in plain.auxiliary_losses
        if operator == "ExpectedValue"
    )
    adjusted_value = next(
        value
        for _identity, operator, value in adjusted.auxiliary_losses
        if operator == "RiskAdjustedExpectedValue"
    )
    assert plain_value != pytest.approx(
        adjusted_value
    )


def test_deferred_loss_statistics_preserve_materialized_metrics():
    outputs, targets = make_outputs_and_targets()

    expected = combined_loss(
        outputs,
        targets,
        MODEL_CONTRACT,
        return_parts=True,
    ).statistics
    deferred = combined_loss(
        outputs,
        targets,
        MODEL_CONTRACT,
        return_statistics=True,
    ).statistics.materialize(torch.tensor(3.25, dtype=torch.float64))

    assert deferred.loss == pytest.approx(expected.loss)
    assert deferred.direct_losses == pytest.approx(expected.direct_losses)
    assert tuple(
        (identity, operator)
        for identity, operator, _value in deferred.auxiliary_losses
    ) == tuple(
        (identity, operator)
        for identity, operator, _value in expected.auxiliary_losses
    )
    assert tuple(
        value for _identity, _operator, value in deferred.auxiliary_losses
    ) == pytest.approx(
        tuple(
            value
            for _identity, _operator, value in expected.auxiliary_losses
        )
    )
    assert deferred.grad_norm == pytest.approx(3.25)


def _expected_value_only_contract() -> ModelContract:
    document = deepcopy(MODEL_CONTRACT.to_document())
    objective = document["objective"]
    assert isinstance(objective, dict)
    objective["resources"] = []
    auxiliary = objective["auxiliaryComponents"]
    assert isinstance(auxiliary, list)
    objective["auxiliaryComponents"] = [
        component
        for component in auxiliary
        if component["operator"] == "ExpectedValue"
    ]
    return ModelContract.from_document(document)


def _weighted_binary_contract(positive_class_weight: float) -> ModelContract:
    document = semantic_fixture_document(
        "positive-class-weighted-binary-w1"
    )["modelContract"]
    assert isinstance(document, dict)
    objective = document["objective"]
    assert isinstance(objective, dict)
    direct = objective["directComponents"]
    assert isinstance(direct, list)
    direct[0]["positiveClassWeight"] = positive_class_weight
    return ModelContract.from_document(document)
