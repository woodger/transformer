import copy
import json
import threading
from dataclasses import asdict

import pytest
import torch
from torch import nn

from app.contracts.worker.v3.config import (
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v3.objective import (
    CHECKPOINT_FORMAT,
    ml_contract,
    objective_config,
)
from app.metrics import TrainMetrics, append_metrics_jsonl, plot_metrics
from app.model.transformer import TransformerModel, public_predictions
from app.storage.checkpoint import load_checkpoint
from app.training.early_stopping import SelectionState
from app.training.factory import build_trainer
from app.training.losses import resolve_loss_stage
from app.training.run_config import model_config_from_args
from app.training.trainer import Trainer
from app.worker.runtime.reproducibility import configure_reproducibility
from app.worker.training import trainer as trainer_module
from app.worker.training.trainer import _BatchPrefetcher


@pytest.fixture(autouse=True)
def torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        yield


def make_dummy_data(n=32, seq_len=5, feat_dim=4):
    source = torch.randn(n, seq_len, feat_dim)
    targets = torch.rand(n, 6)
    targets[:, 0] = torch.rand(n) * 2 - 1
    return source, targets


def model_config(*, seq_len=5, feature_dim=4):
    return ModelConfig(
        seq_len=seq_len,
        hidden=32,
        layers=1,
        dropout=0.0,
        nhead=4,
        feature_dim=feature_dim,
    )


def data_contract(*, seq_len=5, feature_dim=4):
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "dataContractSha256": "d" * 64,
        "seqLen": seq_len,
        "featureDim": feature_dim,
        "targetSchemaId": "inventory.target.v1",
    }


def new_model():
    return TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )


def test_trainer_fit_saves_target_aligned_checkpoint(tmp_path):
    source, targets = make_dummy_data(n=8)
    config = TrainConfig(
        batch_size=4,
        epochs=1,
        loss_schedule="none",
        use_amp=False,
    )
    trainer = build_trainer(
        config,
        new_model(),
        torch.device("cpu"),
        model_config(),
        data_contract=data_contract(),
    )

    path = tmp_path / "model.pth"
    trainer.fit(source, targets, str(path))

    checkpoint = load_checkpoint(str(path), torch.device("cpu"))
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert checkpoint["model_config"]["feature_dim"] == 4
    assert checkpoint["train_config"] == config.to_dict()
    assert checkpoint["data_contract"] == data_contract()
    assert checkpoint["ml_contract"] == ml_contract(config)
    assert checkpoint["objective_config"] == objective_config(config)


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
    source, targets = make_dummy_data(n=4)
    model = new_model()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        epochs=1,
        loss_schedule="none",
        use_amp=True,
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = trainer.fit_batch(source, targets)

    assert metrics.rows == 4
    assert trainer.use_amp is False
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_fit_batch_reports_six_target_metrics():
    source, targets = make_dummy_data()
    trainer = Trainer(
        model=new_model(),
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=8,
        epochs=1,
        loss_schedule="none",
    )

    metrics = trainer.fit_batch(source, targets)

    assert metrics.rows == source.size(0)
    assert metrics.batches == 4
    assert metrics.loss_stage == 4
    for semantic in (
        "mean_return",
        "sigma_return",
        "prob_tp",
        "prob_sl",
        "volatility_next",
        "hitting_prob_tp",
    ):
        assert getattr(metrics, f"{semantic}_mae") >= 0
        assert getattr(metrics, f"{semantic}_rmse") >= 0


def test_predict_batches_model_and_returns_only_public_target_space():
    class RecordingLinear(nn.Linear):
        def __init__(self):
            super().__init__(3, 7)
            self.forward_batch_sizes = []

        def forward(self, inputs):
            self.forward_batch_sizes.append(inputs.size(0))
            return super().forward(inputs)

    model = RecordingLinear()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        epochs=1,
        loss_schedule="none",
    )
    source = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    internal = torch.nn.functional.linear(
        source,
        model.weight.detach(),
        model.bias.detach(),
    )

    predictions = trainer.predict(source)

    assert model.forward_batch_sizes == [4, 4, 2]
    assert torch.allclose(predictions, public_predictions(internal))
    assert predictions.shape == (10, 6)


@pytest.mark.parametrize(
    ("schedule", "progress", "expected"),
    [
        ("epoch", 0, 1),
        ("epoch", 3, 4),
        ("step", 2, 3),
        ("none", 99, 4),
    ],
)
def test_loss_schedule_reaches_target_aligned_maximum_stage(
    schedule,
    progress,
    expected,
):
    assert resolve_loss_stage(
        progress,
        schedule,
        stage_size=1,
        max_stage=4,
    ) == expected


def test_selection_starts_only_after_a_complete_maximum_stage_epoch():
    source, targets = make_dummy_data(n=4)
    selection = CheckpointSelectionConfig(min_delta=0.0, patience=1)
    trainer = Trainer(
        model=new_model(),
        device=torch.device("cpu"),
        lr=0.0,
        batch_size=4,
        epochs=10,
        loss_schedule="epoch",
        stage_size=1,
        selection=selection,
    )

    metrics = trainer.fit_epochs(source, targets)

    assert [item.loss_stage for item in metrics] == [1, 2, 3, 4, 4]
    assert trainer.best_epoch == 4
    assert trainer.selection_state.active is True
    assert trainer.selection_state.wait == 1


def test_selection_disabled_runs_fixed_epochs_and_keeps_last_checkpoint():
    source, targets = make_dummy_data(n=4)
    trainer = Trainer(
        model=new_model(),
        device=torch.device("cpu"),
        lr=0.0,
        batch_size=4,
        epochs=3,
        loss_schedule="none",
    )

    metrics = trainer.fit_epochs(source, targets)

    assert len(metrics) == 3
    assert trainer.best_state_dict is None
    assert trainer.maximum_stage_completed is True


def test_selection_tie_keeps_the_earlier_candidate():
    state = SelectionState(min_delta=0.1, patience=2)
    state.begin()

    assert state.update(1.0) == (True, False)
    assert state.update(0.9) == (False, False)
    assert state.update(0.89) == (True, False)


def test_selection_rejects_nonfinite_score():
    state = SelectionState(min_delta=0.0, patience=1)
    state.begin()

    with pytest.raises(ValueError, match="finite"):
        state.update(float("nan"))


def test_train_metrics_uses_global_row_weighted_direct_losses():
    first = TrainMetrics()
    first.update(
        rows=1,
        loss_parts=_loss_parts(1.0),
        grad_norm=1.0,
        nan_ratio=0.0,
    )
    first.update(
        rows=3,
        loss_parts=_loss_parts(3.0),
        grad_norm=1.0,
        nan_ratio=0.0,
    )

    assert first.direct_losses() == pytest.approx((2.5,) * 6)


def test_closed_batch_prefetch_prepares_exactly_one_batch_ahead():
    second_started = threading.Event()
    third_started = threading.Event()

    def batches():
        yield 0
        second_started.set()
        yield 1
        third_started.set()
        yield 2

    prefetched = _BatchPrefetcher(batches())
    try:
        assert next(prefetched) == 0
        assert second_started.wait(timeout=1.0)
        assert not third_started.wait(timeout=0.05)
        assert next(prefetched) == 1
        assert third_started.wait(timeout=1.0)
        assert next(prefetched) == 2
        with pytest.raises(StopIteration):
            next(prefetched)
    finally:
        prefetched.close()


def test_closed_batch_prefetch_propagates_producer_failure():
    def batches():
        yield 0
        raise RuntimeError("closed replay failed")

    prefetched = _BatchPrefetcher(batches())
    try:
        assert next(prefetched) == 0
        with pytest.raises(RuntimeError, match="closed replay failed"):
            next(prefetched)
    finally:
        prefetched.close()


def test_payload_partitioning_does_not_change_training_state():
    source, targets = make_dummy_data(n=10)
    initial = new_model().state_dict()

    def train(payloads):
        configure_reproducibility(91, deterministic=True)
        model = new_model()
        model.load_state_dict(initial)
        config = TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=2,
            loss_schedule="none",
            seed=91,
            deterministic=True,
        )
        trainer = build_trainer(
            config,
            model,
            torch.device("cpu"),
            model_config(),
        )
        metrics = trainer.fit_payloads(lambda: iter(payloads))
        return model.state_dict(), [
            _semantic_metrics(item) for item in metrics
        ]

    single_state, single_metrics = train(((source, targets),))
    split_state, split_metrics = train((
        (source[:3], targets[:3]),
        (source[3:5], targets[3:5]),
        (source[5:], targets[5:]),
    ))

    assert single_metrics == split_metrics
    assert all(
        torch.equal(single_state[name], split_state[name])
        for name in single_state
    )


def test_closed_and_delayed_streaming_inputs_are_semantically_equivalent(
    monkeypatch,
):
    monkeypatch.setattr(trainer_module, "_MAX_SHUFFLE_WINDOW_BATCHES", 2)
    source, targets = make_dummy_data(n=14)
    initial = copy.deepcopy(new_model().state_dict())

    def train(*, streaming: bool):
        configure_reproducibility(137, deterministic=True)
        model = new_model()
        model.load_state_dict(initial)
        config = TrainConfig(
            lr=1e-3,
            batch_size=4,
            epochs=2,
            loss_schedule="none",
            seed=137,
            deterministic=True,
        )
        trainer = build_trainer(
            config,
            model,
            torch.device("cpu"),
            model_config(),
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
                yield source[:9], targets[:9]
                assert training_started.is_set()
                yield source[9:11], targets[9:11]
                yield source[11:], targets[11:]

            try:
                trainer.fit_streaming_payloads(
                    delayed_first_epoch(),
                    lambda: iter(((source, targets),)),
                    on_epoch=on_epoch,
                )
            finally:
                hook.remove()
        else:
            trainer.fit_payloads(
                lambda: iter(((source, targets),)),
                on_epoch=on_epoch,
            )
        return epochs, trainer.state

    closed_epochs, closed_state = train(streaming=False)
    streaming_epochs, streaming_state = train(streaming=True)

    _assert_nested_equal(closed_epochs, streaming_epochs)
    assert closed_state == streaming_state


def test_metrics_jsonl_contains_per_target_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = TrainMetrics(rows=4, batches=1, loss_l0=0.2)
    append_metrics_jsonl(str(path), metrics, mode="fit")

    row = json.loads(path.read_text().strip())

    assert row["loss_l0"] == pytest.approx(0.2)
    for semantic in (
        "mean_return",
        "sigma_return",
        "prob_tp",
        "prob_sl",
        "volatility_next",
        "hitting_prob_tp",
    ):
        assert f"{semantic}_mae" in row
        assert f"{semantic}_rmse" in row
    assert "ret_mae_skill" not in row


def test_plot_metrics_writes_target_metric_svg(tmp_path):
    path = tmp_path / "metrics.jsonl"
    output = tmp_path / "plots"
    metrics = TrainMetrics(rows=2, mean_return_mae=0.25)
    append_metrics_jsonl(str(path), metrics, mode="fit")

    paths = plot_metrics(str(path), str(output))

    expected = output / "mean_return_mae.svg"
    assert str(expected) in paths
    assert "<svg" in expected.read_text()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_amp_training_updates_parameters():
    source, targets = make_dummy_data(n=4)
    model = new_model().to("cuda")
    trainer = Trainer(
        model=model,
        device=torch.device("cuda"),
        lr=1e-3,
        batch_size=4,
        epochs=1,
        loss_schedule="none",
        use_amp=True,
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = trainer.fit_batch(source, targets)

    assert metrics.rows == 4
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def _loss_parts(value: float) -> dict:
    return {
        "loss": value,
        **{f"loss_l{index}": value for index in range(6)},
        "loss_nll": value,
        "loss_ev": value,
        **{
            f"{semantic}_mae": value
            for semantic in (
                "mean_return",
                "sigma_return",
                "prob_tp",
                "prob_sl",
                "volatility_next",
                "hitting_prob_tp",
            )
        },
        **{
            f"{semantic}_mse": value * value
            for semantic in (
                "mean_return",
                "sigma_return",
                "prob_tp",
                "prob_sl",
                "volatility_next",
                "hitting_prob_tp",
            )
        },
        "loss_stage": 4,
    }


def _semantic_metrics(metrics: TrainMetrics) -> dict:
    result = asdict(metrics)
    for field in (
        "input_pipeline_ms",
        "missing_stats_ms",
        "host_to_device_ms",
        "train_step_ms",
        "elapsed_ms",
    ):
        result.pop(field)
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
