import math
import torch
import json

from torch import nn
from metrics import TrainMetrics, append_metrics_jsonl, plot_metrics
from app.trainer import Trainer
from app.transformer import TransformerModel
from app.utils import MODELS_DIR, resolve_metrics_path


def make_dummy_data(n=32, seq_len=5, feat_dim=4, out_dim=6):
    X = torch.randn(n, seq_len, feat_dim)
    Y = torch.randn(n, out_dim)
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

    assert model_path.exists()


def test_trainer_amp_flag_on_cpu():
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
        use_amp=True,  # просим AMP, но CPU
    )

    assert trainer.use_amp is False


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
    assert metrics.week == 1
    assert metrics.lr == 1e-3
    assert 0.0 <= metrics.nan_ratio <= 1.0
    assert 0.0 <= metrics.valid_token_ratio <= 1.0


def test_trainer_per_week_is_configurable():
    model = nn.Linear(2, 6)

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        per_week=3,
        use_amp=False,
    )

    assert trainer.per_week == 3


def test_trainer_rejects_invalid_per_week():
    model = nn.Linear(2, 6)

    try:
        Trainer(
            model=model,
            device=torch.device("cpu"),
            lr=1e-3,
            batch_size=1,
            epochs=1,
            patience=1,
            per_week=0,
            use_amp=False,
        )
    except ValueError as exc:
        assert "per_week must be a positive integer" in str(exc)
    else:
        raise AssertionError("Trainer accepted invalid per_week")


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
    )

    metrics = trainer.fit_batch(X, Y)
    trainer.record_metrics(metrics, frame=3)

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["frame"] == 3
    assert rows[0]["rows"] == 8
    assert isinstance(rows[0]["loss"], float)
    assert "grad_norm" in rows[0]
    assert "valid_token_ratio" in rows[0]


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


def test_autocast_cpu():
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        use_amp=True,  # юзер просит AMP, но CPU
    )

    # Должен вернуть nullcontext, т.к. CPU
    ctx = trainer._autocast()
    assert ctx.__class__.__name__ == "nullcontext"


def test_autocast_cuda_or_cpu():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        use_amp=True,
    )

    ctx = trainer._autocast()
    if device.type == "cuda":
        # Должен быть torch.amp.autocast
        assert ctx.__class__.__name__ == "autocast"
    else:
        # CPU → nullcontext
        assert ctx.__class__.__name__ == "nullcontext"


def test_autocast_forward_pass_cpu():
    """Проверяем, что с CPU forward можно обернуть _autocast и получить выход"""
    device = torch.device("cpu")
    model = nn.Linear(3, 1)
    trainer = Trainer(model, device, lr=1e-3, batch_size=1, epochs=1, patience=1)
    
    x = torch.randn(5, 3)
    with trainer._autocast():
        y = model(x)
    assert y.shape == (5, 1)


def test_gradscaler_cpu():
    """Проверяем, что GradScaler работает на CPU (по сути не делает масштабирование)"""
    device = torch.device("cpu")
    model = nn.Linear(2, 1)
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        use_amp=True,  # пользователь просит AMP, но CPU
    )

    x = torch.randn(3, 2)
    y = torch.randn(3, 1)
    
    optimizer = trainer.optimizer
    scaler = trainer.scaler

    optimizer.zero_grad()
    with trainer._autocast():  # на CPU → nullcontext
        preds = model(x)
        loss = ((preds - y)**2).mean()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Проверяем, что параметры обновились
    for p in model.parameters():
        assert p.grad is not None

def test_gradscaler_cuda():
    """Проверяем GradScaler на GPU (если есть CUDA)"""
    if not torch.cuda.is_available():
        return  # пропускаем тест на системе без CUDA

    device = torch.device("cuda")
    model = nn.Linear(2, 1).to(device)
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        batch_size=1,
        epochs=1,
        patience=1,
        use_amp=True,
    )

    x = torch.randn(3, 2, device=device)
    y = torch.randn(3, 1, device=device)
    
    optimizer = trainer.optimizer
    scaler = trainer.scaler

    optimizer.zero_grad()
    with trainer._autocast():  # на CUDA → torch.amp.autocast
        preds = model(x)
        loss = ((preds - y)**2).mean()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Проверяем, что параметры обновились
    for p in model.parameters():
        assert p.grad is not None
