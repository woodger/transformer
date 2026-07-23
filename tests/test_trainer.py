import math
import torch
import json

import pytest
from torch import nn
from app.storage.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from app.metrics import TrainMetrics, append_metrics_jsonl, plot_metrics
from app.training.early_stopping import EarlyStopping
from app.training.losses import resolve_loss_stage
from app.training.run_config import ModelConfig, TrainConfig, model_config_from_args
from app.training.trainer import Trainer
from app.model.transformer import TransformerModel
from app.utils import MODELS_DIR, resolve_metrics_path


@pytest.fixture(autouse=True)
def torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        yield


def make_dummy_data(n=32, seq_len=5, feat_dim=4, out_dim=6):
    X = torch.randn(n, seq_len, feat_dim)
    Y = torch.randn(n, out_dim)
    if out_dim >= 6:
        Y[:, 4] = torch.rand(n) + 0.1
        Y[:, 5] = torch.randint(0, 2, (n,), dtype=Y.dtype)
    return X, Y


def test_trainer_fit_cpu(tmp_path):
    X, Y = make_dummy_data()

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=8,
        epochs=2,
        patience=1,
        use_amp=False,
    )

    model_path = tmp_path / "model.pth"
    trainer.fit(X, Y, str(model_path))

    checkpoint = load_checkpoint(str(model_path), torch.device("cpu"))
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert set(checkpoint["state_dict"]) == set(model.state_dict())
    assert all(
        torch.isfinite(value).all()
        for value in checkpoint["state_dict"].values()
    )


def test_trainer_checkpoint_stores_run_config(tmp_path):
    X, Y = make_dummy_data(n=8)
    model_config = ModelConfig(seq_len=5, hidden=32, layers=1, dropout=0.0, nhead=4)
    train_config = TrainConfig(batch_size=4, epochs=1, patience=1, use_amp=False)

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=train_config.lr,
        batch_size=train_config.batch_size,
        epochs=train_config.epochs,
        patience=train_config.patience,
        use_amp=False,
        model_config=model_config,
        train_config=train_config,
    )

    model_path = tmp_path / "model.pth"
    trainer.fit(X, Y, str(model_path))

    checkpoint = load_checkpoint(str(model_path), torch.device("cpu"))
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert checkpoint["model_config"]["seq_len"] == 5
    assert checkpoint["model_config"]["hidden"] == 32
    assert checkpoint["train_config"]["batch_size"] == 4
    assert checkpoint["train_config"]["monitor"] == "ret_mae_skill"


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
    X, Y = make_dummy_data(n=4)

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=8,
        epochs=1,
        patience=1,
        use_amp=True,  # просим AMP, но CPU
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = trainer.fit_batch(X, Y)

    assert metrics.rows == 4
    assert trainer.use_amp is False
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_trainer_fit_batch_cpu():
    X, Y = make_dummy_data()

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=8,
        epochs=1,
        patience=1,
        use_amp=False,
    )

    metrics = trainer.fit_batch(X, Y)

    assert isinstance(metrics, TrainMetrics)
    assert metrics > 0
    assert metrics.rows == X.size(0)
    assert metrics.batches == 4
    assert metrics.step == 4
    assert metrics.loss_stage == 1
    assert metrics.lr == 1e-3
    assert metrics.sigma_min > 0.0
    assert metrics.sigma_p05 > 0.0
    assert metrics.sigma_mean > 0.0
    assert metrics.ret_mae >= 0.0
    assert metrics.ret_rmse >= 0.0
    assert metrics.ret_mae_baseline >= 0.0
    assert 0.0 <= metrics.nan_ratio <= 1.0
    assert 0.0 <= metrics.masked_token_ratio <= 1.0
    assert 0.0 <= metrics.complete_token_ratio <= 1.0
    assert 0.0 <= metrics.partial_token_ratio <= 1.0
    assert 0.0 <= metrics.empty_token_ratio <= 1.0


def test_trainer_stage_size_is_configurable():
    model = nn.Linear(2, 6)

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        stage_size=3,
        use_amp=False,
    )

    assert trainer.stage_size == 3


def test_trainer_loss_schedule_advances_by_epoch():
    X, Y = make_dummy_data(n=5)
    Y[:, 4] = torch.rand(5) + 0.1
    Y[:, 5] = torch.randint(0, 2, (5,), dtype=Y.dtype)

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        loss_schedule="epoch",
        stage_size=1,
        use_amp=False,
    )

    metrics = trainer.fit_batch(X, Y, epoch=3)

    assert metrics.batches == 5
    assert metrics.step == 5
    assert metrics.loss_stage == 4
    assert metrics.loss_prob != 0.0
    assert metrics.loss_ev != 0.0
    assert metrics.loss_vol != 0.0


def test_trainer_loss_schedule_advances_by_optimizer_step():
    X, Y = make_dummy_data(n=5)
    Y[:, 4] = torch.rand(5) + 0.1
    Y[:, 5] = torch.randint(0, 2, (5,), dtype=Y.dtype)

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        loss_schedule="step",
        stage_size=1,
        use_amp=False,
    )

    metrics = trainer.fit_batch(X, Y)

    assert metrics.batches == 5
    assert metrics.step == 5
    assert metrics.loss_stage == 4
    assert metrics.loss_prob != 0.0
    assert metrics.loss_ev != 0.0
    assert metrics.loss_vol != 0.0


def test_trainer_loss_schedule_none_uses_fixed_stage():
    X, Y = make_dummy_data(n=4)

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=2,
        epochs=1,
        patience=1,
        loss_stage=1,
        loss_schedule="none",
        stage_size=1,
        use_amp=False,
    )

    metrics = trainer.fit_batch(X, Y, epoch=10)

    assert metrics.loss_stage == 1
    assert metrics.loss_prob == 0.0
    assert metrics.loss_ev == 0.0
    assert metrics.loss_vol == 0.0


def test_resolve_loss_stage_caps_at_configured_max_stage():
    assert resolve_loss_stage(99, "epoch", stage_size=1, max_stage=3) == 3


def test_trainer_fit_epochs_runs_until_patience_after_full_schedule():
    X, Y = make_dummy_data(n=4)
    Y[:, 4] = torch.rand(4) + 0.1
    Y[:, 5] = torch.randint(0, 2, (4,), dtype=Y.dtype)

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=0.0,
        batch_size=4,
        epochs=8,
        patience=1,
        loss_schedule="epoch",
        stage_size=2,
        monitor="loss",
        use_amp=False,
    )

    seen = []
    metrics_rows = trainer.fit_epochs(
        X,
        Y,
        on_epoch=lambda epoch, metrics, monitor: seen.append((epoch + 1, metrics.loss_stage)),
    )

    assert len(metrics_rows) >= 7
    assert seen[:6] == [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)]
    assert seen[-1][1] == 4


def test_trainer_fit_payloads_runs_global_epochs_over_all_payloads():
    first = make_dummy_data(n=3)
    second = make_dummy_data(n=2)
    for _, targets in (first, second):
        targets[:, 4] = torch.rand(targets.size(0)) + 0.1
        targets[:, 5] = torch.randint(
            0,
            2,
            (targets.size(0),),
            dtype=targets.dtype,
        )

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=2,
        epochs=3,
        patience=0,
        loss_schedule="epoch",
        stage_size=1,
        monitor="loss",
        use_amp=False,
    )
    payload_passes = 0

    def payloads():
        nonlocal payload_passes
        payload_passes += 1
        return iter((first, second))

    metrics_rows = trainer.fit_payloads(payloads)

    assert payload_passes == 3
    assert [metrics.rows for metrics in metrics_rows] == [5, 5, 5]
    assert [metrics.batches for metrics in metrics_rows] == [3, 3, 3]
    assert [metrics.step for metrics in metrics_rows] == [3, 6, 9]
    assert [metrics.loss_stage for metrics in metrics_rows] == [1, 2, 3]
    assert trainer.best_frame is None


def test_trainer_fit_payloads_uses_one_early_stopper_for_the_job():
    X = torch.zeros(2, 5, 4)
    Y = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    ])
    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=0.0,
        batch_size=1,
        epochs=5,
        patience=1,
        loss_schedule="none",
        monitor="loss",
        use_amp=False,
    )

    metrics_rows = trainer.fit_payloads(
        lambda: iter(((X[:1], Y[:1]), (X[1:], Y[1:])))
    )

    assert len(metrics_rows) == 2
    assert [metrics.rows for metrics in metrics_rows] == [2, 2]
    assert [metrics.step for metrics in metrics_rows] == [2, 4]


def test_trainer_rejects_invalid_stage_size():
    model = nn.Linear(2, 6)

    try:
        Trainer(
            model=model,
            device=torch.device("cpu"),
            lr=1e-3,
            batch_size=1,
            epochs=1,
            patience=1,
            stage_size=0,
            use_amp=False,
        )
    except ValueError as exc:
        assert "stage_size must be a positive integer" in str(exc)
    else:
        raise AssertionError("Trainer accepted invalid stage_size")


def test_resolve_metrics_path_uses_models_dir():
    assert resolve_metrics_path(None) is None
    assert resolve_metrics_path("train.jsonl") == f"{MODELS_DIR}/train.jsonl"


def test_trainer_writes_metrics_jsonl(tmp_path):
    X, Y = make_dummy_data(n=8)
    metrics_path = tmp_path / "models" / "metrics.jsonl"

    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        epochs=1,
        patience=1,
        use_amp=False,
        metrics_path=str(metrics_path),
        metrics_context={
            "hidden": 32,
            "layers": 1,
            "seq_len": 5,
        },
    )

    metrics = trainer.fit_batch(X, Y)
    trainer.record_metrics(metrics, frame=3)

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["frame"] == 3
    assert rows[0]["rows"] == 8
    assert isinstance(rows[0]["loss"], float)
    assert "grad_norm" in rows[0]
    assert rows[0]["step"] == 2
    assert "loss_stage" in rows[0]
    assert "sigma_min" in rows[0]
    assert "sigma_p05" in rows[0]
    assert "sigma_mean" in rows[0]
    assert "ret_mae" in rows[0]
    assert "ret_rmse" in rows[0]
    assert "ret_mae_baseline" in rows[0]
    assert "ret_mae_skill" in rows[0]
    assert "ret_mae_improvement" in rows[0]
    assert "masked_token_ratio" in rows[0]
    assert "complete_token_ratio" in rows[0]
    assert "partial_token_ratio" in rows[0]
    assert "empty_token_ratio" in rows[0]
    assert rows[0]["context_mode"] == "relaxed"
    assert rows[0]["batch_size"] == 4
    assert rows[0]["loss_schedule"] == "epoch"
    assert rows[0]["stage_size"] == 5
    assert rows[0]["max_loss_stage"] == 4
    assert rows[0]["hidden"] == 32
    assert rows[0]["layers"] == 1
    assert rows[0]["seq_len"] == 5
    assert rows[0]["device"] == "cpu"
    assert rows[0]["monitor"] == "ret_mae_skill"
    assert rows[0]["monitor_min_improvement"] == 0.0


def test_trainer_log_line_includes_run_config():
    X, Y = make_dummy_data(n=4)
    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        epochs=1,
        patience=1,
        use_amp=False,
        metrics_context={
            "hidden": 32,
            "layers": 1,
            "seq_len": 5,
        },
    )

    metrics = trainer.fit_batch(X, Y)
    output = metrics.log_line(epoch=1, **trainer.metrics_context)

    assert "batch_size=4" in output
    assert "loss_schedule=epoch" in output
    assert "stage_size=5" in output
    assert "max_loss_stage=4" in output
    assert "hidden=32" in output
    assert "layers=1" in output
    assert "seq_len=5" in output
    assert "device=cpu" in output
    assert "ret_mae=" in output
    assert "ret_rmse=" in output
    assert "ret_mae_baseline=" in output
    assert "ret_mae_skill=" in output
    assert "ret_mae_improvement=" in output


def test_metrics_jsonl_serializes_nonfinite_as_null(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics = TrainMetrics(
        rows=1,
        batches=1,
        loss=math.inf,
        grad_norm=math.nan,
    )

    append_metrics_jsonl(str(metrics_path), metrics, frame=1)

    row = json.loads(metrics_path.read_text())
    assert row["loss"] is None
    assert row["grad_norm"] is None


def test_train_metrics_aggregates_return_errors():
    metrics = TrainMetrics()

    metrics.update(
        rows=1,
        loss_parts={
            "loss": 0.0,
            "loss_ret": 0.0,
            "loss_prob": 0.0,
            "loss_ev": 0.0,
            "loss_vol": 0.0,
            "ret_mae": 1.0,
            "ret_mse": 1.0,
            "ret_mae_baseline": 2.0,
        },
        grad_norm=0.0,
        nan_ratio=0.0,
    )
    metrics.update(
        rows=3,
        loss_parts={
            "loss": 0.0,
            "loss_ret": 0.0,
            "loss_prob": 0.0,
            "loss_ev": 0.0,
            "loss_vol": 0.0,
            "ret_mae": 3.0,
            "ret_mse": 9.0,
            "ret_mae_baseline": 4.0,
        },
        grad_norm=0.0,
        nan_ratio=0.0,
    )

    assert metrics.ret_mae == 2.5
    assert metrics.ret_rmse == math.sqrt(7.0)
    assert metrics.ret_mae_baseline == 3.5
    assert metrics.ret_mae_skill == 2.5 / 3.5
    assert metrics.ret_mae_improvement == 1.0 - (2.5 / 3.5)


def test_trainer_monitor_requires_baseline_improvement():
    trainer = Trainer(
        model=nn.Linear(1, 6),
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        monitor="ret_mae_skill",
        monitor_min_improvement=0.01,
        use_amp=False,
    )

    worse = TrainMetrics(ret_mae=1.0, ret_mae_baseline=1.0, ret_mae_skill=1.0)
    better = TrainMetrics(ret_mae=0.98, ret_mae_baseline=1.0, ret_mae_skill=0.98)

    assert trainer._baseline_passed(worse) is False
    assert trainer._baseline_passed(better) is True


def test_early_stopping_tracks_monitor_before_baseline_passes():
    stopper = EarlyStopping(patience=2, min_stage=1)

    assert stopper.update(1.50, stage=1) is False
    assert stopper.update(1.40, stage=1) is False
    assert stopper.update(1.30, stage=1) is False
    assert stopper.wait == 0

    assert stopper.update(1.31, stage=1) is False
    assert stopper.update(1.32, stage=1) is True


def test_trainer_save_restores_best_monitored_checkpoint(tmp_path):
    model = nn.Linear(1, 6)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        monitor="ret_mae_skill",
        use_amp=False,
    )

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)

    best = TrainMetrics(ret_mae=0.5, ret_mae_baseline=1.0, ret_mae_skill=0.5)
    payload = trainer._observe_metrics(best, frame=1, epoch=1)
    assert payload["checkpoint_best"] is True

    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(2.0)

    model_path = tmp_path / "best.pth"
    trainer.save(str(model_path))

    checkpoint = load_checkpoint(str(model_path), torch.device("cpu"))
    assert checkpoint["extra"]["checkpoint_selection"]["source"] == "best_monitor"
    assert torch.all(checkpoint["state_dict"]["weight"] == 1.0)
    assert torch.all(checkpoint["state_dict"]["bias"] == 1.0)


def test_plot_metrics_writes_svg(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join([
            json.dumps({"frame": 1, "loss": 2.0, "grad_norm": 1.5}),
            json.dumps({"frame": 2, "loss": 1.0, "grad_norm": 1.1}),
        ])
    )
    plots_dir = tmp_path / "plots"

    paths = plot_metrics(str(metrics_path), str(plots_dir))

    assert str(plots_dir / "loss.svg") in paths
    assert str(plots_dir / "grad_norm.svg") in paths
    assert (plots_dir / "loss.svg").read_text().startswith("<svg")


def test_plot_metrics_skips_nonfinite_values(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join([
            json.dumps({"frame": 1, "loss": None, "grad_norm": None}),
            json.dumps({"frame": 2, "loss": 1.0, "grad_norm": 1.1}),
        ])
    )
    plots_dir = tmp_path / "plots"

    paths = plot_metrics(str(metrics_path), str(plots_dir))

    assert str(plots_dir / "loss.svg") in paths
    assert str(plots_dir / "grad_norm.svg") in paths


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device is required for AMP training integration",
)
def test_cuda_amp_training_updates_parameters():
    device = torch.device("cuda")
    X, Y = make_dummy_data(n=4)
    model = TransformerModel(
        input_dim=4,
        seq_len=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
    ).to(device)
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        batch_size=4,
        epochs=1,
        patience=1,
        use_amp=True,
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = trainer.fit_batch(X, Y)

    assert metrics.rows == 4
    assert trainer.use_amp is True
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
