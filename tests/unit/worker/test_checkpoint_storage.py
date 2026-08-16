import os

import pytest
import torch

import app.worker.checkpoints.model as checkpoint_module
from app.contracts.worker.v5.config import ModelConfig, TrainConfig
from app.worker.checkpoints.model import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    model_path,
    save_checkpoint,
)


def test_relative_model_path_cannot_escape_models_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint_module, "MODELS_DIR", str(tmp_path / "models"))

    with pytest.raises(ValueError, match="must stay inside models"):
        model_path("../outside.pth")


def test_absolute_model_path_remains_supported(tmp_path):
    path = tmp_path / "nested" / "model.pth"

    assert model_path(path) == str(path)


def test_relative_model_path_cannot_escape_through_symlink(monkeypatch, tmp_path):
    models = tmp_path / "models"
    outside = tmp_path / "outside"
    models.mkdir()
    outside.mkdir()
    (models / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(checkpoint_module, "MODELS_DIR", str(models))

    with pytest.raises(ValueError, match="must stay inside models"):
        model_path("linked/model.pth")


def test_checkpoint_save_atomically_replaces_existing_file(tmp_path):
    path = tmp_path / "nested" / "model.pth"
    model = torch.nn.Linear(2, 1)
    path.parent.mkdir()
    path.write_bytes(b"old checkpoint")

    save_checkpoint(
        path,
        model,
        model_config=ModelConfig(seq_len=1, feature_dim=2),
        train_config=TrainConfig(),
    )

    checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert set(checkpoint["state_dict"]) == {"weight", "bias"}
    assert not [name for name in os.listdir(path.parent) if name.endswith(".tmp")]


def test_unknown_wrapped_checkpoint_format_is_rejected(tmp_path):
    path = tmp_path / "model.pth"
    torch.save(
        {"format": "transformer-checkpoint-v999", "state_dict": {}},
        path,
    )

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_checkpoint(path, torch.device("cpu"))
