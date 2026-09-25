import copy
import math
import threading
import warnings
from contextlib import nullcontext
from dataclasses import asdict

import pytest
import torch
from torch import nn

import app.worker.training.trainer as trainer_module
from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v17.config import (
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v17.diagnostics import DiagnosticsConfig
from app.contracts.worker.v17.model_definition import resolved_semantic_digests
from app.worker.data.tensors import TrainingBatch
from app.worker.model.transformer import TransformerModel, public_predictions
from app.worker.runtime.reproducibility import configure_reproducibility
from app.worker.telemetry import (
    EpochTelemetry,
    ObservedTrainingEpoch,
    epoch_telemetry_document,
)
from app.worker.training import batching as batching_module
from app.worker.training.batching import BatchPrefetcher
from app.worker.training.early_stopping import SelectionState
from app.worker.training.epoch import TrainingEpochResult
from app.worker.training.factory import build_trainer
from app.worker.training.losses import MaterializedLossStatistics
from app.worker.training.run_config import model_config_from_args
from app.worker.training.trainer import Trainer
from tests.fixture_documents import semantic_fixture_document


def _contract(
    fixture: str,
    *,
    hidden: int,
    layers: int,
    dropout: float,
    nhead: int,
    mode: str,
) -> ModelContract:
    document = semantic_fixture_document(fixture)["modelContract"]
    assert isinstance(document, dict)
    tuning = document["modelTuning"]
    assert isinstance(tuning, dict)
    tuning.update({
        "hiddenWidth": hidden,
        "encoderLayerCount": layers,
        "dropoutProbability": dropout,
        "attentionHeadCount": nhead,
        "missingValuePolicy": mode,
    })
    return ModelContract.from_document(document)


DEFAULT_MODEL_CONTRACT = _contract(
    "multi-target-shared-resource",
    hidden=32,
    layers=1,
    dropout=0.0,
    nhead=4,
    mode="relaxed",
)
TARGETS = DEFAULT_MODEL_CONTRACT.target_identities
DIRECT_COMPONENTS = tuple(
    (str(item["identity"]), str(item["operator"]))
    for item in DEFAULT_MODEL_CONTRACT.direct_components
)
AUXILIARY_COMPONENTS = tuple(
    (str(item["identity"]), str(item["operator"]))
    for item in DEFAULT_MODEL_CONTRACT.auxiliary_components
)


@pytest.fixture(autouse=True)
def torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        yield


def make_dummy_data(n=32, seq_len=5, feat_dim=4):
    features = torch.randn(n, seq_len, feat_dim)
    targets = torch.rand(n, DEFAULT_MODEL_CONTRACT.target_width)
    return TrainingBatch(features=features, targets=targets)


def slice_batch(batch, rows):
    return TrainingBatch(
        features=batch.features[rows],
        targets=batch.targets[rows],
    )


def model_config(*, seq_len=5, feature_dim=4):
    return ModelConfig(
        seq_len=seq_len,
        hidden=32,
        layers=1,
        dropout=0.0,
        nhead=4,
        feature_dim=feature_dim,
    )


def _model_definition_sha256(contract: ModelContract) -> str:
    return str(resolved_semantic_digests(
        contract,
        "d" * 64,
        model_config(),
    )["modelDefinitionSha256"])


def new_model(contract=DEFAULT_MODEL_CONTRACT):
    return TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        model_contract=contract,
        nhead=4,
    )


def new_trainer(
    train_config: TrainConfig,
    *,
    model: nn.Module | None = None,
    contract=DEFAULT_MODEL_CONTRACT,
) -> Trainer:
    return Trainer(
        model=new_model(contract) if model is None else model,
        device=torch.device("cpu"),
        train_config=train_config,
        model_contract=contract,
        model_config=model_config(),
        model_definition_sha256=_model_definition_sha256(contract),
    )


def test_model_config_can_be_loaded_from_checkpoint_defaults():
    class Args:
        seq_len = None
        hidden = None
        layers = None
        dropout = None
        nhead = None
        context_mode = None
        out_dim = None

    config = model_config_from_args(
        Args(),
        checkpoint_config={
            "seq_len": 12,
            "feature_dim": 48,
            "hidden": 512,
            "layers": 4,
            "context_mode": "relaxed",
        },
    )

    assert config.seq_len == 12
    assert config.hidden == 512
    assert config.layers == 4
    assert config.context_mode == "relaxed"


def test_cpu_training_disables_amp_and_updates_parameters():
    batch = make_dummy_data(n=4)
    model = new_model()
    trainer = new_trainer(
        TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=1,
            use_amp=True,
        ),
        model=model,
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = trainer.fit_batch(batch)

    assert metrics.rows == 4
    assert trainer.use_amp is False
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_fit_batch_reports_target_metrics():
    batch = make_dummy_data()
    trainer = new_trainer(
        TrainConfig(
            lr=1e-3,
            batch_size=8,
            epochs=1,
        ),
    )

    metrics = trainer.fit_batch(batch)

    assert metrics.rows == batch.features.size(0)
    assert metrics.batches == 4
    telemetry = metrics.telemetry
    assert telemetry is not None
    assert telemetry.training_batches_completed == 4
    assert telemetry.optimizer_updates_applied == 4
    assert telemetry.optimizer_updates_skipped == 0
    assert telemetry.amp_overflow_batches == 0
    assert telemetry.finite_gradient_batches == 4
    assert telemetry.non_finite_gradient_batches == 0
    for target in TARGETS:
        assert telemetry.target_mae[target] >= 0
        assert telemetry.target_rmse[target] >= 0


def test_gradient_interactions_are_sampled_at_completed_step_interval():
    batch = make_dummy_data(n=4)
    trainer = new_trainer(
        TrainConfig(
            lr=1e-3,
            batch_size=2,
            epochs=1,
            diagnostics=DiagnosticsConfig(
                gradient_every_steps=2,
            ),
        ),
    )

    metrics = trainer.fit_batch(batch)

    assert metrics.telemetry is not None
    assert metrics.telemetry.gradient_interaction_samples == 1
    assert metrics.telemetry.gradient_interactions_document() is not None


def test_single_target_objective_trains_and_predicts_one_public_value():
    contract = _contract(
        "single-regression",
        hidden=32,
        layers=1,
        dropout=0.0,
        nhead=4,
        mode="relaxed",
    )
    full_batch = make_dummy_data(n=4)
    batch = TrainingBatch(
        features=full_batch.features,
        targets=full_batch.targets[:, :1],
    )
    trainer = new_trainer(
        TrainConfig(lr=1e-3, batch_size=4, epochs=1),
        contract=contract,
    )

    metrics = trainer.fit_batch(batch)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        predictions = trainer.predict(batch.features)

    assert metrics.direct_loss_values.keys() == {"direct.mean-return"}
    assert metrics.auxiliary_loss_values == {}
    assert predictions.shape == (4, 1)
    assert not any(
        "nested tensors is in prototype stage" in str(item.message)
        for item in emitted
    )


def test_target_error_telemetry_failure_does_not_interrupt_training(
    monkeypatch,
    capsys,
):
    class BrokenTargetErrorObservation:
        @staticmethod
        def evaluate(*_args):
            raise RuntimeError("injected telemetry failure")

    monkeypatch.setattr(
        trainer_module,
        "TargetErrorObservation",
        BrokenTargetErrorObservation,
    )
    batch = make_dummy_data(n=4)
    trainer = new_trainer(
        TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=1,
        ),
    )

    result = trainer.fit_batch(batch)

    assert result.rows == 4
    assert result.batches == 1
    assert result.telemetry is None
    assert "training epoch telemetry disabled" in capsys.readouterr().err


def test_gradient_diagnostics_failure_does_not_interrupt_training(
    monkeypatch,
    capsys,
):
    class BrokenGradientInteractionObservation:
        @staticmethod
        def evaluate(*_args):
            raise RuntimeError("injected diagnostics failure")

    monkeypatch.setattr(
        trainer_module,
        "GradientInteractionObservation",
        BrokenGradientInteractionObservation,
    )
    batch = make_dummy_data(n=4)
    trainer = new_trainer(
        TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=1,
            diagnostics=DiagnosticsConfig(
                gradient_every_steps=1,
            ),
        ),
    )

    result = trainer.fit_batch(batch)

    assert result.rows == 4
    assert result.batches == 1
    assert result.telemetry is None
    assert "training epoch telemetry disabled" in capsys.readouterr().err


def test_predict_batches_model_and_returns_only_public_target_space():
    class RecordingLinear(nn.Linear):
        def __init__(self):
            super().__init__(3, 4)
            self.forward_batch_sizes = []

        def forward(self, inputs):
            self.forward_batch_sizes.append(inputs.size(0))
            return super().forward(inputs)

    model = RecordingLinear()
    trainer = new_trainer(
        TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=1,
        ),
        model=model,
    )
    source = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    internal = torch.nn.functional.linear(
        source,
        model.weight.detach(),
        model.bias.detach(),
    )

    predictions = trainer.predict(source)

    assert model.forward_batch_sizes == [4, 4, 2]
    assert torch.allclose(
        predictions,
        public_predictions(internal, DEFAULT_MODEL_CONTRACT),
    )
    assert predictions.shape == (10, 3)


def test_selection_starts_with_the_first_complete_epoch():
    batch = make_dummy_data(n=4)
    selection = CheckpointSelectionConfig(min_delta=0.0, patience=1)
    trainer = new_trainer(
        TrainConfig(
            lr=1e-30,
            batch_size=4,
            epochs=10,
            selection=selection,
        ),
    )

    metrics = trainer.fit_epochs(batch)

    assert len(metrics) == 2
    assert trainer.best_epoch == 1
    assert trainer.selection_state is not None
    assert trainer.selection_state.active is True
    assert trainer.selection_state.wait == 1


def test_selection_disabled_runs_fixed_epochs_and_keeps_last_checkpoint():
    batch = make_dummy_data(n=4)
    trainer = new_trainer(
        TrainConfig(
            lr=1e-30,
            batch_size=4,
            epochs=3,
        ),
    )

    metrics = trainer.fit_epochs(batch)

    assert len(metrics) == 3
    assert trainer.best_state_dict is None


def test_selection_tie_keeps_the_earlier_candidate():
    state = SelectionState(min_delta=0.1, patience=2)
    state.begin()

    first = state.update(1.0)
    tie = state.update(0.9)
    improved = state.update(0.89)

    assert first.improved is True
    assert first.should_stop is False
    assert tie.improved is False
    assert tie.should_stop is False
    assert improved.improved is True
    assert improved.should_stop is False


def test_selection_rejects_nonfinite_score():
    state = SelectionState(min_delta=0.0, patience=1)
    state.begin()

    with pytest.raises(ValueError, match="finite"):
        state.update(float("nan"))


def test_train_metrics_uses_global_row_weighted_direct_losses():
    first = TrainingEpochResult(
        targets=TARGETS,
        direct_components=DIRECT_COMPONENTS,
    )
    first.update(
        rows=1,
        statistics=_statistics(1.0),
        step=1,
    )
    first.update(
        rows=3,
        statistics=_statistics(3.0),
        step=2,
    )

    assert first.direct_losses() == pytest.approx((2.5,) * len(TARGETS))


def test_nonfinite_gradient_does_not_discard_finite_epoch_statistics():
    metrics = ObservedTrainingEpoch(
        targets=TARGETS,
        direct_components=DIRECT_COMPONENTS,
        telemetry=EpochTelemetry(targets=TARGETS),
    )
    for gradient, applied, overflow in (
        (1.0, True, False),
        (float("inf"), False, True),
        (3.0, True, False),
        (2.0, True, False),
    ):
        metrics.update(
            rows=1,
            statistics=_statistics(1.0),
            step=metrics.step + 1,
        )
        assert metrics.telemetry is not None
        metrics.telemetry.observe_batch(
            rows=1,
            target_errors=_target_errors(1.0),
            grad_norm=gradient,
            optimizer_update_applied=applied,
            amp_overflow=overflow,
            nan_ratio=0.0,
        )

    document = epoch_telemetry_document(metrics)
    assert document is not None

    assert document["trainingBatchesCompleted"] == 4
    assert document["optimizerUpdatesApplied"] == 3
    assert document["optimizerUpdatesSkipped"] == 1
    assert document["ampOverflowBatches"] == 1
    assert document["finiteGradientBatches"] == 3
    assert document["nonFiniteGradientBatches"] == 1
    assert document["preClipGradientNormMean"] == pytest.approx(2.0)
    assert document["preClipGradientNormMax"] == pytest.approx(3.0)
    assert document["preClipGradientNormP95"] == pytest.approx(3.0)


def test_invalid_gradient_telemetry_does_not_interrupt_metric_aggregation():
    metrics = ObservedTrainingEpoch(
        targets=TARGETS,
        direct_components=DIRECT_COMPONENTS,
        telemetry=EpochTelemetry(targets=TARGETS),
    )

    metrics.update(
        rows=1,
        statistics=_statistics(1.0),
        step=1,
    )
    assert metrics.telemetry is not None
    metrics.telemetry.observe_batch(
        rows=1,
        target_errors=_target_errors(1.0),
        grad_norm=-1.0,
        optimizer_update_applied=True,
        amp_overflow=True,
        nan_ratio=0.0,
    )
    document = epoch_telemetry_document(metrics)
    assert document is not None

    assert document["trainingBatchesCompleted"] == 1
    assert document["optimizerUpdatesApplied"] == 1
    assert document["ampOverflowBatches"] == 1
    assert document["finiteGradientBatches"] == 0
    assert document["nonFiniteGradientBatches"] == 1
    assert document["preClipGradientNormMean"] is None


def test_amp_overflow_counts_a_skipped_update_and_keeps_later_gradient():
    batch = make_dummy_data(n=4)
    model = new_model()
    trainer = new_trainer(
        TrainConfig(
            batch_size=2,
            epochs=1,
        ),
        model=model,
    )
    forward_calls = 0

    def count_forward(*_args):
        nonlocal forward_calls
        forward_calls += 1

    forward_hook = model.register_forward_hook(count_forward)
    gradient_hooks = [
        parameter.register_hook(
            lambda gradient: (
                torch.full_like(gradient, torch.inf)
                if forward_calls == 1
                else gradient
            )
        )
        for parameter in model.parameters()
    ]
    trainer.use_amp = True
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    trainer._autocast = nullcontext
    try:
        metrics = trainer.fit_batch(batch)
    finally:
        forward_hook.remove()
        for hook in gradient_hooks:
            hook.remove()

    telemetry = metrics.telemetry
    assert telemetry is not None
    assert telemetry.training_batches_completed == 2
    assert telemetry.optimizer_updates_applied == 1
    assert telemetry.optimizer_updates_skipped == 1
    assert telemetry.amp_overflow_batches == 1
    assert telemetry.finite_gradient_batches == 1
    assert telemetry.non_finite_gradient_batches == 1
    assert telemetry.pre_clip_gradient_norm_mean is not None
    assert math.isfinite(telemetry.pre_clip_gradient_norm_mean)


def test_closed_batch_prefetch_prepares_exactly_one_batch_ahead():
    second_started = threading.Event()
    third_started = threading.Event()

    def batches():
        yield make_dummy_data(n=1)
        second_started.set()
        yield make_dummy_data(n=2)
        third_started.set()
        yield make_dummy_data(n=3)

    prefetched = BatchPrefetcher(batches())
    try:
        assert next(prefetched).features.size(0) == 1
        assert second_started.wait(timeout=1.0)
        assert not third_started.wait(timeout=0.05)
        assert next(prefetched).features.size(0) == 2
        assert third_started.wait(timeout=1.0)
        assert next(prefetched).features.size(0) == 3
        with pytest.raises(StopIteration):
            next(prefetched)
    finally:
        prefetched.close()


def test_closed_batch_prefetch_propagates_producer_failure():
    def batches():
        yield make_dummy_data(n=1)
        raise RuntimeError("closed replay failed")

    prefetched = BatchPrefetcher(batches())
    try:
        assert next(prefetched).features.size(0) == 1
        with pytest.raises(RuntimeError, match="closed replay failed"):
            next(prefetched)
    finally:
        prefetched.close()


def test_payload_partitioning_does_not_change_training_state():
    batch = make_dummy_data(n=10)
    initial = new_model().state_dict()

    def train(payloads):
        configure_reproducibility(91, deterministic=True)
        model = new_model()
        model.load_state_dict(initial)
        config = TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=2,
            seed=91,
            deterministic=True,
        )
        trainer = build_trainer(
            config,
            model,
        torch.device("cpu"),
        model_config(),
        model_contract=DEFAULT_MODEL_CONTRACT,
        model_definition_sha256=_model_definition_sha256(
            DEFAULT_MODEL_CONTRACT,
        ),
        )
        metrics = trainer.fit_payloads(lambda: iter(payloads))
        return model.state_dict(), [
            _semantic_metrics(item) for item in metrics
        ]

    single_state, single_metrics = train((batch,))
    split_state, split_metrics = train((
        slice_batch(batch, slice(None, 3)),
        slice_batch(batch, slice(3, 5)),
        slice_batch(batch, slice(5, None)),
    ))

    assert single_metrics == split_metrics
    assert all(
        torch.equal(single_state[name], split_state[name])
        for name in single_state
    )


def test_closed_and_delayed_streaming_inputs_are_semantically_equivalent(
    monkeypatch,
):
    monkeypatch.setattr(batching_module, "_MAX_SHUFFLE_WINDOW_BATCHES", 2)
    batch = make_dummy_data(n=14)
    initial = copy.deepcopy(new_model().state_dict())

    def train(*, streaming: bool):
        configure_reproducibility(137, deterministic=True)
        model = new_model()
        model.load_state_dict(initial)
        config = TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=2,
            seed=137,
            deterministic=True,
        )
        trainer = build_trainer(
            config,
            model,
        torch.device("cpu"),
        model_config(),
        model_contract=DEFAULT_MODEL_CONTRACT,
        model_definition_sha256=_model_definition_sha256(
            DEFAULT_MODEL_CONTRACT,
        ),
        )
        epochs = []

        def on_epoch(_epoch, metrics, _selection):
            epochs.append({
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
                "metrics": _semantic_metrics(metrics),
            })

        if streaming:
            training_started = threading.Event()
            hook = model.register_forward_pre_hook(
                lambda *_args: training_started.set()
            )

            def delayed_first_epoch():
                yield slice_batch(batch, slice(None, 9))
                assert training_started.is_set()
                yield slice_batch(batch, slice(9, 11))
                yield slice_batch(batch, slice(11, None))

            try:
                trainer.fit_streaming_payloads(
                    delayed_first_epoch(),
                    lambda: iter((batch,)),
                    on_epoch=on_epoch,
                )
            finally:
                hook.remove()
        else:
            trainer.fit_payloads(
                lambda: iter((batch,)),
                on_epoch=on_epoch,
            )
        return epochs, trainer.state

    closed_epochs, closed_state = train(streaming=False)
    streaming_epochs, streaming_state = train(streaming=True)

    _assert_nested_equal(closed_epochs, streaming_epochs)
    assert closed_state == streaming_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_amp_recovers_scale_and_updates_parameters():
    batch = make_dummy_data(n=4)
    model = new_model().to("cuda")
    trainer = Trainer(
        model=model,
        device=torch.device("cuda"),
        train_config=TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=1,
            use_amp=True,
        ),
        model_contract=DEFAULT_MODEL_CONTRACT,
        model_definition_sha256=_model_definition_sha256(
            DEFAULT_MODEL_CONTRACT,
        ),
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    for epoch in range(32):
        metrics = trainer.fit_batch(batch, epoch=epoch)
        telemetry = metrics.telemetry

        assert metrics.rows == 4
        assert telemetry is not None
        assert telemetry.training_batches_completed == 1
        assert (
            telemetry.optimizer_updates_applied
            + telemetry.optimizer_updates_skipped
        ) == 1
        if telemetry.optimizer_updates_applied == 1:
            break
    else:
        pytest.fail("CUDA AMP scale did not recover after 32 training batches")

    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def _statistics(value: float) -> MaterializedLossStatistics:
    return MaterializedLossStatistics(
        loss=value,
        direct_losses=(value,) * len(TARGETS),
        auxiliary_losses=(),
        grad_norm=None,
    )


def _target_errors(value: float) -> dict[str, tuple[float, float]]:
    return {
        target: (value, value * value)
        for target in TARGETS
    }


def _semantic_metrics(metrics: TrainingEpochResult) -> dict:
    result = asdict(metrics)
    telemetry = result["telemetry"]
    if telemetry is None:
        return result
    for field in (
        "input_pipeline_ms",
        "missing_stats_ms",
        "host_to_device_ms",
        "train_step_ms",
        "elapsed_ms",
    ):
        telemetry.pop(field)
    return result


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right
