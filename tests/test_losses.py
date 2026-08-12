import pytest
import torch

from app.training.losses import combined_loss


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
def test_maximum_stage_directly_supervises_every_public_head(target_index):
    outputs, targets = make_outputs_and_targets()

    loss = combined_loss(outputs, targets, loss_stage=4)
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

    _, original = combined_loss(
        outputs,
        targets,
        loss_stage=4,
        return_parts=True,
    )
    _, changed = combined_loss(
        outputs,
        changed_targets,
        loss_stage=4,
        return_parts=True,
    )

    assert changed[f"loss_l{target_index}"] != pytest.approx(
        original[f"loss_l{target_index}"]
    )
    for other_index in set(range(6)) - {target_index}:
        assert changed[f"loss_l{other_index}"] == pytest.approx(
            original[f"loss_l{other_index}"]
        )


def test_sigma_return_and_private_gaussian_scale_are_distinct_heads():
    outputs, targets = make_outputs_and_targets()

    loss = combined_loss(outputs, targets, loss_stage=1)
    loss.backward()

    assert outputs.grad[0, 1].abs() > 0
    assert outputs.grad[0, 6].abs() > 0


def test_probability_targets_use_logits_for_stable_direct_loss():
    outputs, targets = make_outputs_and_targets()

    _, parts = combined_loss(
        outputs,
        targets,
        loss_stage=4,
        return_parts=True,
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs[:, 2],
        targets[:, 2],
    )
    assert parts["loss_l2"] == pytest.approx(expected.item())


def test_deferred_loss_statistics_preserve_materialized_metrics():
    outputs, targets = make_outputs_and_targets()

    _, expected = combined_loss(
        outputs,
        targets,
        loss_stage=4,
        return_parts=True,
    )
    _, statistics = combined_loss(
        outputs,
        targets,
        loss_stage=4,
        return_statistics=True,
    )
    actual, grad_norm = statistics.materialize(
        torch.tensor(3.25, dtype=torch.float64)
    )

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value)
    assert grad_norm == pytest.approx(3.25)
