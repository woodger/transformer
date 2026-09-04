import pytest
import torch

from app.contracts.worker.v11.objective import ObjectiveConfig, default_objective
from app.worker.training.losses import combined_loss


def make_outputs_and_targets():
    outputs = torch.tensor(
        [[0.1, 0.3, -0.4, 0.7, 0.2, -0.2, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[-0.2, 0.7, 0.8, 0.1, 0.6, 0.9]],
        dtype=torch.float64,
    )
    return outputs, targets


@pytest.mark.parametrize("target_index", range(6))
def test_declarative_objective_directly_supervises_every_selected_head(
    target_index,
):
    outputs, targets = make_outputs_and_targets()

    loss = combined_loss(outputs, targets, default_objective())
    loss.backward()

    gradient = outputs.grad[0, target_index]
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0


@pytest.mark.parametrize(
    ("target_index", "replacement"),
    [(0, 0.6), (1, 0.2), (2, 0.1), (3, 0.8), (4, 0.2), (5, 0.2)],
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
        default_objective(),
        return_parts=True,
    ).statistics
    changed = combined_loss(
        outputs,
        changed_targets,
        default_objective(),
        return_parts=True,
    ).statistics

    assert changed.direct_losses[target_index] != pytest.approx(
        original.direct_losses[target_index]
    )
    for other_index in set(range(6)) - {target_index}:
        assert changed.direct_losses[other_index] == pytest.approx(
            original.direct_losses[other_index]
        )


def test_sigma_return_and_private_gaussian_scale_are_distinct_heads():
    outputs, targets = make_outputs_and_targets()

    loss = combined_loss(outputs, targets, default_objective())
    loss.backward()

    assert outputs.grad[0, 1].abs() > 0
    assert outputs.grad[0, 6].abs() > 0


def test_probability_targets_use_logits_for_stable_direct_loss():
    outputs, targets = make_outputs_and_targets()

    statistics = combined_loss(
        outputs,
        targets,
        default_objective(),
        return_parts=True,
    ).statistics

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs[:, 2],
        targets[:, 2],
    )
    assert statistics.direct_losses[2] == pytest.approx(expected.item())


def test_expected_value_and_risk_adjusted_expected_value_are_distinct():
    outputs, targets = make_outputs_and_targets()
    expected_value = _objective(
        auxiliary=[{"operator": "ExpectedValue", "weight": 0.3}],
    )

    plain = combined_loss(
        outputs[:, :6],
        targets,
        expected_value,
        return_parts=True,
    ).statistics
    adjusted = combined_loss(
        outputs,
        targets,
        default_objective(),
        return_parts=True,
    ).statistics

    assert plain.auxiliary_losses[0][0] == "ExpectedValue"
    assert adjusted.auxiliary_losses[1][0] == "RiskAdjustedExpectedValue"
    assert plain.auxiliary_losses[0][1] != pytest.approx(
        adjusted.auxiliary_losses[1][1]
    )


def test_deferred_loss_statistics_preserve_materialized_metrics():
    outputs, targets = make_outputs_and_targets()

    expected = combined_loss(
        outputs,
        targets,
        default_objective(),
        return_parts=True,
    ).statistics
    deferred = combined_loss(
        outputs,
        targets,
        default_objective(),
        return_statistics=True,
    ).statistics.materialize(torch.tensor(3.25, dtype=torch.float64))

    assert deferred.loss == pytest.approx(expected.loss)
    assert deferred.direct_losses == pytest.approx(expected.direct_losses)
    assert tuple(name for name, _value in deferred.auxiliary_losses) == tuple(
        name for name, _value in expected.auxiliary_losses
    )
    assert tuple(
        value for _name, value in deferred.auxiliary_losses
    ) == pytest.approx(
        tuple(value for _name, value in expected.auxiliary_losses)
    )
    assert deferred.grad_norm == pytest.approx(3.25)


def _objective(*, auxiliary: list[dict[str, object]]) -> ObjectiveConfig:
    document = default_objective().to_document()
    objective = document["objective"]
    assert isinstance(objective, dict)
    objective["auxiliaryLosses"] = auxiliary
    return ObjectiveConfig.from_document(document)
