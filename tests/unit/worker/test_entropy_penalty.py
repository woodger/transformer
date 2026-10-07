import math
from copy import deepcopy

import pytest
import torch

from app.contracts.checkpoint.v14 import CHECKPOINT_FORMAT
from app.contracts.metrics.v13 import build_training_record, project_training_points
from app.contracts.semantic.v7 import ModelContract
from app.contracts.worker.v22.codec import validate_training_metrics_for_model
from app.contracts.worker.v22.config import (
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v22.model_definition import resolved_semantic_digests
from app.worker.data.tensors import TrainingBatch
from app.worker.telemetry.epoch import epoch_telemetry_document
from app.worker.telemetry.gradient_interactions import GradientInteractionObservation
from app.worker.training.losses import combined_loss
from app.worker.training.trainer import Trainer
from tests.fixture_documents import semantic_fixture_document


def _contract(*, weighted=False, coefficient=0.1, direct_weight=1.0, class_weight=28.0):
    fixture = (
        "weighted-binary-entropy-penalty" if weighted else "bernoulli-entropy-penalty"
    )
    document = semantic_fixture_document(fixture)["modelContract"]
    document["objective"]["directComponents"][0]["weight"] = direct_weight
    if weighted:
        document["objective"]["directComponents"][0]["positiveClassWeight"] = (
            class_weight
        )
    document["objective"]["auxiliaryComponents"][0]["weight"] = coefficient
    return ModelContract.from_document(document)


def _penalty(logits, contract):
    evaluation = combined_loss(
        logits,
        torch.zeros_like(logits),
        contract,
        return_statistics=True,
    )
    return dict(evaluation.diagnostic_components)["auxiliary.entropy"]


@pytest.mark.parametrize(
    ("logit", "entropy"),
    [
        (-10000.0, 0.0),
        (-math.log(3), -0.25 * math.log(0.25) - 0.75 * math.log(0.75)),
        (0.0, math.log(2)),
        (math.log(3), -0.25 * math.log(0.25) - 0.75 * math.log(0.75)),
        (10000.0, 0.0),
    ],
)
def test_bernoulli_penalties_have_opposite_signs_and_expected_entropy(logit, entropy):
    document = _contract(coefficient=1.0).to_document()
    logits = torch.tensor([[logit]], dtype=torch.float64)

    positive = _penalty(logits, ModelContract.from_document(document))
    document["objective"]["auxiliaryComponents"][0]["operator"] = (
        "BernoulliConfidencePenalty"
    )
    negative = _penalty(logits, ModelContract.from_document(document))

    assert positive.item() == pytest.approx(entropy, abs=1e-12)
    assert negative.item() == pytest.approx(-entropy, abs=1e-12)
    assert 0 <= positive.item() <= math.log(2) + 1e-12


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("direct_weight", [0.5, 2.0])
@pytest.mark.parametrize("coefficient", [0.01, 0.1, 1.0])
def test_entropy_penalty_adds_weighted_mean_entropy_without_changing_bce(
    weighted,
    direct_weight,
    coefficient,
):
    contract = _contract(
        weighted=weighted, coefficient=coefficient, direct_weight=direct_weight
    )
    correction = math.log(28) if weighted else 0
    logits = (
        torch.tensor([[-math.log(3)], [0.0], [math.log(3)]], dtype=torch.float64)
        + correction
    )
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64)
    entropy = (-0.5 * math.log(0.25) - 1.5 * math.log(0.75) + math.log(2)) / 3
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(28.0) if weighted else None,
    )
    baseline = deepcopy(contract.to_document())
    baseline["objective"].pop("auxiliaryComponents")

    evaluation = combined_loss(logits, targets, contract, return_parts=True)
    direct_only = combined_loss(logits, targets, ModelContract.from_document(baseline))

    assert evaluation.statistics.direct_losses == pytest.approx((bce.item(),))
    assert evaluation.statistics.auxiliary_losses[0][2] == pytest.approx(entropy)
    assert evaluation.loss.item() == pytest.approx(
        direct_weight * bce.item() + coefficient * entropy
    )
    assert direct_only.item() == pytest.approx(direct_weight * bce.item())


def test_entropy_penalty_gradient_moves_probabilities_away_from_one_half():
    contract = _contract(coefficient=1.0)
    logits = torch.tensor(
        [[-4.0], [0.0], [4.0]], dtype=torch.float64, requires_grad=True
    )
    penalty = _penalty(logits, contract)

    gradient = torch.autograd.grad(penalty, logits)[0]
    after_step = logits.detach() - 0.1 * gradient

    assert gradient[0, 0] > 0
    assert gradient[1, 0] == 0
    assert gradient[2, 0] < 0
    assert _penalty(after_step, contract).item() < penalty.item()


@pytest.mark.parametrize("class_weight", [1.0, 28.0])
def test_weighted_entropy_penalty_matches_value_and_gradient_at_the_public_logit(
    class_weight,
):
    ordinary = _contract(coefficient=1.0)
    weighted = _contract(weighted=True, coefficient=1.0, class_weight=class_weight)
    public_logits = torch.tensor(
        [[-4.0], [0.0], [4.0]], dtype=torch.float64, requires_grad=True
    )
    raw_logits = (public_logits.detach() + math.log(class_weight)).requires_grad_()

    ordinary_penalty = _penalty(public_logits, ordinary)
    weighted_penalty = _penalty(raw_logits, weighted)
    ordinary_gradient = torch.autograd.grad(ordinary_penalty, public_logits)[0]
    weighted_gradient = torch.autograd.grad(weighted_penalty, raw_logits)[0]

    torch.testing.assert_close(
        weighted_penalty, ordinary_penalty, rtol=1e-12, atol=1e-14
    )
    torch.testing.assert_close(
        weighted_gradient, ordinary_gradient, rtol=1e-12, atol=1e-14
    )
    maximum = _penalty(
        torch.tensor([[math.log(class_weight)]], dtype=torch.float64), weighted
    )
    assert maximum.item() == pytest.approx(math.log(2))
    if class_weight > 1:
        assert (
            _penalty(torch.zeros(1, 1, dtype=torch.float64), weighted).item()
            < maximum.item()
        )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_entropy_penalty_preserves_finite_loss_and_gradients_at_extreme_logits(
    weighted, dtype
):
    contract = _contract(weighted=weighted)
    logits = torch.tensor(
        [[-10000.0], [0.0], [10000.0]], dtype=dtype, requires_grad=True
    )
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32)

    evaluation = combined_loss(logits, targets, contract, return_statistics=True)
    evaluation.loss.backward()

    assert torch.isfinite(evaluation.loss)
    assert torch.isfinite(logits.grad).all()
    assert 0 <= evaluation.statistics.auxiliary_losses[0][2].item() <= math.log(2)


@pytest.mark.parametrize("weighted", [False, True])
def test_entropy_penalty_gradients_match_finite_differences(weighted):
    contract = _contract(weighted=weighted)
    logits = torch.tensor(
        [[-4.0], [1.0], [6.0]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda values: _penalty(values, contract), (logits,)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_entropy_penalty_trains_with_cuda_amp():
    contract = _contract(weighted=True)
    with torch.random.fork_rng():
        model = torch.nn.Linear(2, 1).cuda()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.GradScaler("cuda")
        features = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], device="cuda")
        targets = torch.tensor([[0.0], [1.0]], device="cuda")
        before = model.weight.detach().clone()

        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = combined_loss(model(features), targets, contract)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not torch.equal(before, model.weight):
                break

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert not torch.equal(before, model.weight)


def test_entropy_penalty_gradient_diagnostics_use_its_weighted_contribution():
    contract = _contract(coefficient=0.2)
    logits = torch.tensor([[4.0]], dtype=torch.float64, requires_grad=True)
    evaluation = combined_loss(
        logits, torch.zeros_like(logits), contract, return_statistics=True
    )

    observation = GradientInteractionObservation.evaluate(
        evaluation.diagnostic_components, logits
    ).materialize()

    probability = 1 / (1 + math.exp(-4))
    components = {
        item["componentIdentity"]: item["norm"] for item in observation["components"]
    }
    assert components["auxiliary.entropy"] == pytest.approx(
        0.2 * 4 * probability * (1 - probability)
    )
    assert observation["pairs"][0]["cosine"] == pytest.approx(-1.0)


def test_entropy_penalty_preserves_direct_selection_and_publishes_positive_auxiliary_telemetry():
    contract = _contract(coefficient=2.0)
    config = ModelConfig.from_tuning(contract.model_tuning, seq_len=2, feature_dim=2)
    digests = resolved_semantic_digests(contract, "a" * 64, config)
    batch = TrainingBatch(
        features=torch.zeros(2, 2, 2), targets=torch.tensor([[0.0], [1.0]])
    )

    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 1))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        trainer = Trainer(
            model=model,
            device=torch.device("cpu"),
            train_config=TrainConfig(
                batch_size=2,
                epochs=1,
                weight_decay=0.0,
                selection=CheckpointSelectionConfig(min_delta=0.0, patience=0),
            ),
            model_contract=contract,
            model_config=config,
            model_definition_sha256=digests["modelDefinitionSha256"],
        )
        result = trainer.fit_epochs(batch)[0]

    assert result.loss == pytest.approx(3 * math.log(2))
    assert result.selection_score == pytest.approx(math.log(2))
    assert trainer.best_selection_score == pytest.approx(math.log(2))
    telemetry = epoch_telemetry_document(
        result,
        mode="fit-stream",
        frame=0,
        epoch=1,
        checkpointBest=True,
        shouldStop=False,
        bestSelectionScore=trainer.best_selection_score,
    )
    validate_training_metrics_for_model(telemetry, contract)
    assert telemetry["auxiliaryLosses"][0]["operator"] == "BernoulliEntropyPenalty"
    assert telemetry["auxiliaryLosses"][0]["value"] == pytest.approx(math.log(2))

    record = build_training_record(
        telemetry,
        recorded_at=0,
        job_id="11111111-1111-4111-8111-111111111111",
        attempt_id="22222222-2222-4222-8222-222222222222",
        attempt=1,
        model_ref="mdl_11111111111111111111111111111111",
        semantic_digests=digests,
        checkpoint_format=CHECKPOINT_FORMAT,
        application_version="0.1.22",
        git_commit="a" * 40,
        targets=contract.target_identities,
    )
    points = project_training_points(record, deployment_id="dev-0")
    auxiliary = next(
        point
        for point in points
        if point["metric"]["name"] == "training.loss.auxiliary"
    )
    assert auxiliary["component"]["operator"] == "BernoulliEntropyPenalty"
    assert auxiliary["metric"]["value"] == pytest.approx(math.log(2))
