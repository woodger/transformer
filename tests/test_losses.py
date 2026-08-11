import math

import pytest
import torch

from app.training.losses import combined_loss

VARIANCE_EPS = 1e-6


def make_stage_one_inputs(mean, sigma, target_mean):
    zeros = torch.zeros_like(mean)
    preds = torch.stack((mean, sigma, zeros, zeros, zeros, zeros), dim=1)
    targets = torch.stack(
        (target_mean, zeros, zeros, zeros, zeros, zeros),
        dim=1,
    )
    return preds, targets


def test_return_loss_matches_gaussian_nll_with_stabilized_variance():
    mean = torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64)
    sigma = torch.tensor([0.25, 0.5, 1.5], dtype=torch.float64)
    target_mean = torch.tensor([0.3, -0.1, -0.2], dtype=torch.float64)
    preds, targets = make_stage_one_inputs(mean, sigma, target_mean)

    loss = combined_loss(preds, targets, loss_stage=1)

    variance = sigma.square() + VARIANCE_EPS
    expected = torch.mean(
        0.5 * ((target_mean - mean).square() / variance + torch.log(variance))
    )
    assert loss == pytest.approx(expected.item())


def test_return_loss_and_gradients_are_finite_for_tiny_sigma():
    mean = torch.tensor([0.0, 0.2], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([1e-12, 1e-9], dtype=torch.float64, requires_grad=True)
    target_mean = torch.tensor([0.1, -0.1], dtype=torch.float64)
    preds, targets = make_stage_one_inputs(mean, sigma, target_mean)

    loss = combined_loss(preds, targets, loss_stage=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(sigma.grad).all()


def test_zero_residual_sigma_collapse_pressure_vanishes_below_variance_floor():
    log_sigma = torch.tensor(math.log(1e-6), dtype=torch.float64, requires_grad=True)
    sigma = log_sigma.exp().reshape(1)
    mean = torch.zeros(1, dtype=torch.float64)
    preds, targets = make_stage_one_inputs(mean, sigma, mean.clone())

    loss = combined_loss(preds, targets, loss_stage=1)
    loss.backward()

    floor_loss = 0.5 * math.log(VARIANCE_EPS)
    assert loss.item() == pytest.approx(floor_loss, abs=1e-6)
    assert abs(log_sigma.grad.item()) < 2e-6


def test_deferred_loss_statistics_preserve_materialized_metrics():
    mean = torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64)
    sigma = torch.tensor([0.25, 0.5, 1.5], dtype=torch.float64)
    target_mean = torch.tensor([0.3, -0.1, -0.2], dtype=torch.float64)
    preds, targets = make_stage_one_inputs(mean, sigma, target_mean)

    _, expected = combined_loss(
        preds,
        targets,
        loss_stage=1,
        return_parts=True,
    )
    _, statistics = combined_loss(
        preds,
        targets,
        loss_stage=1,
        return_statistics=True,
    )
    actual, grad_norm = statistics.materialize(
        torch.tensor(3.25, dtype=torch.float64)
    )

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value)
    assert grad_norm == pytest.approx(3.25)
