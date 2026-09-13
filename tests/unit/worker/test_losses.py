from copy import deepcopy

import pytest
import torch

from app.contracts.semantic.v2 import ModelContract
from app.worker.training.losses import combined_loss
from tests.support.consumer_neutral import model_contract

MODEL_CONTRACT = model_contract("multi-target-shared-resource")


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
