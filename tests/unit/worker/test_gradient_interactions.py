import pytest
import torch

from app.worker.telemetry.gradient_interactions import (
    GradientInteractionObservation,
)


def test_gradient_interaction_observation_reports_norms_and_pairwise_cosine():
    shared = torch.tensor([[1.0, 2.0]], requires_grad=True)
    components = (
        ("target:MeanReturn", 2.0 * shared[0, 0]),
        ("target:SigmaReturn", 3.0 * shared[0, 1]),
    )

    observation = GradientInteractionObservation.evaluate(components, shared)
    document = observation.materialize()

    assert document["components"] == [
        {"name": "target:MeanReturn", "norm": 2.0},
        {"name": "target:SigmaReturn", "norm": 3.0},
    ]
    assert document["pairs"][0]["left"] == "target:MeanReturn"
    assert document["pairs"][0]["right"] == "target:SigmaReturn"
    assert document["pairs"][0]["cosine"] == pytest.approx(0.0)
