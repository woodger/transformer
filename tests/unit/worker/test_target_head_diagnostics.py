from contextlib import nullcontext

import pytest
import torch

from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v18 import validate_document
from app.worker.model.transformer import TransformerModel, public_predictions
from app.worker.telemetry.target_head import TargetHeadDiagnosticsCollector
from app.worker.training.losses import combined_loss
from tests.fixture_documents import semantic_fixture_document


def test_target_head_diagnostics_collects_direct_gradient_and_corrected_prediction():
    document = semantic_fixture_document(
        "positive-class-weighted-binary-w28"
    )
    model_contract = document["modelContract"]
    assert isinstance(model_contract, dict)
    contract = ModelContract.from_document(model_contract)
    model = TransformerModel(
        input_dim=8,
        seq_len=2,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
        context_mode="strict",
    )
    features = torch.randn(4, 2, 8)
    targets = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    collector = TargetHeadDiagnosticsCollector(
        contract,
        collect_encoder_learning=True,
    )
    collector.begin_epoch(model)

    evaluation = combined_loss(
        model(features),
        targets,
        contract,
        return_statistics=True,
    )
    direct_component = evaluation.diagnostic_components[0][1]
    head = model.head.target_head
    weight_gradient, bias_gradient = torch.autograd.grad(
        direct_component,
        (head.weight, head.bias),
        retain_graph=True,
    )
    expected_gradient_l2 = torch.sqrt(
        weight_gradient[0].square().sum() + bias_gradient[0].square()
    ).item()

    collector.observe_component_gradients(
        evaluation.diagnostic_components,
        model,
    )
    evaluation.loss.backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    collector.snapshot_encoder_parameters(model)
    optimizer.step()
    collector.observe_encoder_parameter_updates(
        model,
        optimizer_update_applied=True,
    )

    model.train()
    collector.observe_post_update(
        model,
        lambda: (features[:2], features[2:]),
        device=torch.device("cpu"),
        autocast=nullcontext,
        epoch=1,
        global_step=1,
        expected_rows=4,
    )
    artifact = collector.artifact(
        job_id="11111111-1111-4111-8111-111111111111",
        attempt=1,
        attempt_id="22222222-2222-4222-8222-222222222222",
        input_revision=1,
        manifest_sha256="a" * 64,
        model_definition_sha256="b" * 64,
        job_config_sha256="c" * 64,
        completed_epochs=1,
    )

    assert artifact is not None
    validate_document(artifact, "target-head-diagnostics-artifact")
    assert model.training

    epoch = artifact["epochs"][0]
    assert isinstance(epoch, dict)
    target_head = epoch["targetHeads"][0]
    assert isinstance(target_head, dict)
    assert target_head["gradientBatchCount"] == 1
    assert target_head["gradientL2Mean"] == pytest.approx(
        expected_gradient_l2,
    )
    assert target_head["gradientL2Maximum"] == pytest.approx(
        expected_gradient_l2,
    )

    model.eval()
    with torch.no_grad():
        model_output, shared = model.forward_with_shared_representation(features)
        (
            diagnostic_output,
            diagnostic_shared,
            encoder_input,
            encoder_layers,
        ) = model.forward_with_representation_flow(features)
        raw_logits = model_output[:, 0]
        predictions = public_predictions(model_output, contract)[:, 0]

    torch.testing.assert_close(diagnostic_output, model_output)
    torch.testing.assert_close(diagnostic_shared, shared)

    representation_flow = epoch["representationFlow"]
    assert isinstance(representation_flow, dict)
    encoder_input_summary = representation_flow["encoderInput"]
    encoder_layers_summary = representation_flow["encoderLayers"]
    target_head_input_summary = representation_flow["targetHeadInput"]
    assert isinstance(encoder_input_summary, dict)
    assert isinstance(encoder_layers_summary, list)
    assert isinstance(target_head_input_summary, dict)
    assert len(encoder_layers_summary) == 1
    assert isinstance(encoder_layers_summary[0], dict)

    def row_centered_l2_mean(values: torch.Tensor) -> float:
        centered = values.double() - values.double().mean(dim=0)
        return centered.square().sum(dim=1).sqrt().mean().item()

    assert encoder_input_summary["rowCenteredL2Mean"] == pytest.approx(
        row_centered_l2_mean(encoder_input),
        abs=1e-6,
    )
    assert encoder_layers_summary[0]["layerIndex"] == 0
    assert encoder_layers_summary[0]["rowCenteredL2Mean"] == pytest.approx(
        row_centered_l2_mean(encoder_layers[0]),
        abs=1e-6,
    )
    assert target_head_input_summary["rowCenteredL2Mean"] == pytest.approx(
        row_centered_l2_mean(shared),
        abs=1e-6,
    )

    raw_summary = target_head["rawLogit"]
    public_summary = target_head["publicPrediction"]
    assert isinstance(raw_summary, dict)
    assert isinstance(public_summary, dict)
    assert raw_summary["mean"] == pytest.approx(
        raw_logits.double().mean().item(),
        abs=1e-6,
    )
    assert raw_summary["standardDeviation"] == pytest.approx(
        raw_logits.double().std(unbiased=False).item(),
        abs=1e-6,
    )
    assert public_summary["mean"] == pytest.approx(
        predictions.double().mean().item(),
        abs=1e-6,
    )
    assert public_summary["standardDeviation"] == pytest.approx(
        predictions.double().std(unbiased=False).item(),
        abs=1e-6,
    )
