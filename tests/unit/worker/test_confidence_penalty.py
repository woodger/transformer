import math

import pytest
import torch

from app.contracts.semantic.v6 import ModelContract
from app.contracts.worker.v21.codec import validate_training_metrics_for_model
from app.contracts.worker.v21.config import (
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v21.model_definition import resolved_semantic_digests
from app.worker.data.tensors import TrainingBatch
from app.worker.telemetry.epoch import epoch_telemetry_document
from app.worker.training.losses import combined_loss
from app.worker.training.trainer import Trainer
from tests.fixture_documents import semantic_fixture_document


def _contract(*, weighted=False, coefficient=0.1):
    fixture = (
        "weighted-binary-confidence-penalty"
        if weighted
        else "bernoulli-confidence-penalty"
    )
    document = semantic_fixture_document(fixture)["modelContract"]
    document["objective"]["auxiliaryComponents"][0]["weight"] = coefficient
    return ModelContract.from_document(document)


@pytest.mark.parametrize("coefficient", [0.01, 0.1, 1.0])
def test_confidence_penalty_subtracts_mean_entropy_without_changing_bce(coefficient):
    contract = _contract(coefficient=coefficient)
    outputs = torch.tensor([[-math.log(3)], [0.0], [math.log(3)]], dtype=torch.float64)
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64)
    probabilities = (0.25, 0.5, 0.75)
    entropy = sum(
        -p * math.log(p) - (1 - p) * math.log(1 - p)
        for p in probabilities
    ) / len(probabilities)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

    statistics = combined_loss(
        outputs,
        targets,
        contract,
        return_parts=True,
    ).statistics

    assert statistics.direct_losses == pytest.approx((bce.item(),))
    assert statistics.auxiliary_losses[0][2] == pytest.approx(-entropy)
    assert statistics.loss == pytest.approx(bce.item() - coefficient * entropy)


def test_confidence_penalty_gradient_moves_probabilities_toward_one_half():
    contract = _contract()
    baseline = contract.to_document()
    baseline["objective"].pop("auxiliaryComponents")
    outputs = torch.tensor([[-4.0], [0.0], [4.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64)

    regularized_loss = combined_loss(outputs, targets, contract)
    baseline_loss = combined_loss(outputs, targets, ModelContract.from_document(baseline))
    gradient = torch.autograd.grad(regularized_loss - baseline_loss, outputs)[0]

    assert gradient[0, 0] < 0
    assert gradient[1, 0] == 0
    assert gradient[2, 0] > 0


def test_weighted_binary_penalty_uses_the_corrected_public_probability():
    contract = _contract(weighted=True)
    outputs = torch.tensor([[0.0], [math.log(28)]], dtype=torch.float64)
    targets = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    public_probability = 1 / 29
    entropy_at_zero = (
        -public_probability * math.log(public_probability)
        - (1 - public_probability) * math.log(1 - public_probability)
    )

    statistics = combined_loss(
        outputs,
        targets,
        contract,
        return_parts=True,
    ).statistics

    assert statistics.auxiliary_losses[0][2] == pytest.approx(
        -(entropy_at_zero + math.log(2)) / 2,
    )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_confidence_penalty_preserves_finite_loss_and_gradients_at_extreme_logits(
    weighted,
    dtype,
):
    contract = _contract(weighted=weighted)
    outputs = torch.tensor([[-10000.0], [0.0], [10000.0]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32)

    evaluation = combined_loss(outputs, targets, contract, return_statistics=True)
    evaluation.loss.backward()

    assert torch.isfinite(evaluation.loss)
    assert torch.isfinite(outputs.grad).all()
    assert torch.isfinite(evaluation.statistics.auxiliary_losses[0][2])


@pytest.mark.parametrize("weighted", [False, True])
def test_confidence_penalty_gradients_match_finite_differences(weighted):
    contract = _contract(weighted=weighted)
    outputs = torch.tensor([[-4.0], [1.0], [6.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64)

    assert torch.autograd.gradcheck(
        lambda values: combined_loss(values, targets, contract),
        (outputs,),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_confidence_penalty_trains_with_cuda_amp():
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

        # Weighted BCE gradients can overflow until GradScaler calibrates its scale.
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


def test_negative_regularized_loss_preserves_direct_loss_selection_and_telemetry():
    contract = _contract(coefficient=2.0)
    config = ModelConfig.from_tuning(contract.model_tuning, seq_len=2, feature_dim=2)
    batch = TrainingBatch(
        features=torch.zeros(2, 2, 2),
        targets=torch.tensor([[0.0], [1.0]]),
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
            model_definition_sha256=resolved_semantic_digests(
                contract,
                "a" * 64,
                config,
            )["modelDefinitionSha256"],
        )
        result = trainer.fit_epochs(batch)[0]

    assert result.loss == pytest.approx(-math.log(2))
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
    assert telemetry is not None
    validate_training_metrics_for_model(telemetry, contract)
    assert telemetry["auxiliaryLosses"][0]["value"] == pytest.approx(-math.log(2))
