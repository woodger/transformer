import pytest
import torch

from app.worker.telemetry.gradient_interactions import (
    GradientInteractionObservation,
)


def test_gradient_interaction_observation_reports_norms_and_pairwise_cosine():
    shared = torch.tensor([[1.0, 2.0]], requires_grad=True)
    components = (
        ("direct.first", 2.0 * shared[0, 0]),
        ("direct.second", 3.0 * shared[0, 1]),
    )

    observation = GradientInteractionObservation.evaluate(components, shared)
    document = observation.materialize()

    assert document["components"] == [
        {"componentIdentity": "direct.first", "norm": 2.0},
        {"componentIdentity": "direct.second", "norm": 3.0},
    ]
    assert document["pairs"][0]["leftComponentIdentity"] == "direct.first"
    assert document["pairs"][0]["rightComponentIdentity"] == "direct.second"
    assert document["pairs"][0]["cosine"] == pytest.approx(0.0)
