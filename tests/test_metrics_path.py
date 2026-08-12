import pytest

import app.worker.utils as utils_module
from app.worker.utils import resolve_metrics_path


def test_omitted_metrics_path_remains_disabled():
    assert resolve_metrics_path(None) is None


def test_relative_metrics_path_stays_inside_models(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(utils_module, "MODELS_DIR", str(models_dir))

    assert resolve_metrics_path("runs/train.jsonl") == str(
        models_dir / "runs" / "train.jsonl"
    )


def test_relative_metrics_path_cannot_escape_models(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_module, "MODELS_DIR", str(tmp_path / "models"))

    with pytest.raises(ValueError, match="must stay inside models"):
        resolve_metrics_path("../train.jsonl")
