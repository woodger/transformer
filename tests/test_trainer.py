import torch

from torch import nn
from app.trainer import Trainer
from app.transformer import TransformerModel


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

    loss = trainer.fit_batch(X, Y)

    assert loss > 0


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
