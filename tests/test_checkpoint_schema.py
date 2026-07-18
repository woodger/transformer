import importlib
import io
from types import SimpleNamespace

import pytest
import torch

from app.storage.checkpoint import load_checkpoint_metadata, save_checkpoint
from app.training.run_config import ModelConfig


fit_module = importlib.import_module("app.commands.fit")
fit_stream_module = importlib.import_module("app.commands.fit_stream")
predict_module = importlib.import_module("app.commands.predict")
predict_stream_module = importlib.import_module("app.commands.predict_stream")


def model_args(**overrides):
    values = {
        "seq_len": 2,
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
        "nhead": 2,
        "context_mode": "relaxed",
        "out_dim": 6,
        "model_name": "model.pth",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def checkpoint_model_config(feature_dim):
    return ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        context_mode="relaxed",
        feature_dim=feature_dim,
    ).to_dict()


def test_checkpoint_metadata_describes_training_data_schema(tmp_path):
    path = tmp_path / "model.pth"
    model_config = ModelConfig(
        seq_len=3,
        hidden=8,
        layers=1,
        dropout=0.0,
        nhead=2,
        context_mode="relaxed",
        feature_dim=4,
    )

    save_checkpoint(path, torch.nn.Linear(2, 1), model_config=model_config)

    metadata = load_checkpoint_metadata(path)
    assert metadata["data_schema"] == {
        "schema_version": 1,
        "tensor_dtype": "float32",
        "src": {
            "column": "src",
            "accepted_element_types": ["float32", "float64"],
            "width": 12,
        },
        "tgt": {
            "column": "tgt",
            "accepted_element_types": ["float32", "float64"],
            "width": 6,
        },
        "feature_dim": 4,
        "model_input_feature_dim": 8,
        "context_mode": "relaxed",
        "normalization": None,
        "missing": {"nan_fill": 0.0, "flags": "per-feature"},
    }


def test_v1_checkpoint_metadata_without_data_schema_remains_supported(tmp_path):
    path = tmp_path / "model-v1.pth"
    torch.save(
        {
            "format": "transformer-checkpoint-v1",
            "state_dict": {},
            "model_config": checkpoint_model_config(feature_dim=None),
        },
        path,
    )

    metadata = load_checkpoint_metadata(path)

    assert metadata["data_schema"] is None


def test_file_fit_freezes_actual_feature_dim_before_building(monkeypatch):
    args = model_args(data="train.arrow")
    captured = {}

    monkeypatch.setattr(
        fit_module,
        "read_arrow",
        lambda _: (torch.randn(2, 6), torch.randn(2, 6)),
    )

    def build_model(config, *args):
        captured["model_config"] = config
        return object()

    class Trainer:
        def fit(self, *args):
            pass

    def build_trainer(args, model, device, config):
        captured["trainer_config"] = config
        return Trainer()

    fit_module.run(
        args,
        torch.device("cpu"),
        build_model_fn=build_model,
        build_trainer_fn=build_trainer,
    )

    assert captured["model_config"].feature_dim == 3
    assert captured["trainer_config"].feature_dim == 3


def test_file_fit_rejects_empty_training_input(monkeypatch):
    args = model_args(data="train.arrow")
    monkeypatch.setattr(
        fit_module,
        "read_arrow",
        lambda _: (torch.empty((0, 0)), torch.empty((0, 0))),
    )

    with pytest.raises(ValueError, match="Training input contains no rows"):
        fit_module.run(args, torch.device("cpu"))


def test_stream_fit_freezes_first_frame_feature_dim_before_building(monkeypatch):
    args = model_args()
    table = SimpleNamespace(num_rows=1)
    captured = {}

    monkeypatch.setattr(fit_stream_module.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(
        fit_stream_module,
        "iter_framed_arrow",
        lambda *args, **kwargs: iter([table]),
    )
    monkeypatch.setattr(
        fit_stream_module,
        "table_to_tensors",
        lambda _: (torch.randn(1, 4), torch.randn(1, 6)),
    )

    def build_model(config, *args):
        captured["model_config"] = config
        return object()

    class Trainer:
        def fit_epochs(self, *args, **kwargs):
            pass

        def save(self, *args):
            pass

    def build_trainer(args, model, device, config):
        captured["trainer_config"] = config
        return Trainer()

    fit_stream_module.run(
        args,
        torch.device("cpu"),
        build_model_fn=build_model,
        build_trainer_fn=build_trainer,
    )

    assert captured["model_config"].feature_dim == 2
    assert captured["trainer_config"].feature_dim == 2


def test_file_predict_rejects_feature_dim_conflicting_with_checkpoint(monkeypatch):
    args = model_args(
        data="input.arrow",
        preds_path="predictions.arrow",
        seq_len=None,
        hidden=None,
        layers=None,
        dropout=None,
        nhead=None,
        context_mode=None,
        out_dim=None,
    )
    monkeypatch.setattr(
        predict_module,
        "load_checkpoint_metadata",
        lambda *args: {"model_config": checkpoint_model_config(feature_dim=2)},
    )
    monkeypatch.setattr(
        predict_module,
        "read_source_arrow",
        lambda _: torch.randn(1, 6),
    )

    with pytest.raises(
        ValueError,
        match="Feature dim 3 does not match checkpoint feature_dim 2",
    ):
        predict_module.run(args, torch.device("cpu"))


def test_file_predict_writes_typed_empty_output_without_building_model(monkeypatch):
    args = model_args(
        data="input.arrow",
        preds_path="predictions.arrow",
        pred_col="out",
        seq_len=None,
        hidden=None,
        layers=None,
        dropout=None,
        nhead=None,
        context_mode=None,
        out_dim=None,
    )
    written = {}
    monkeypatch.setattr(
        predict_module,
        "load_checkpoint_metadata",
        lambda *args: {"model_config": checkpoint_model_config(feature_dim=2)},
    )
    monkeypatch.setattr(
        predict_module,
        "read_source_arrow",
        lambda _: torch.empty((0, 0)),
    )
    monkeypatch.setattr(
        predict_module,
        "write_arrow",
        lambda path, preds, col: written.update(path=path, preds=preds, col=col),
    )

    predict_module.run(
        args,
        torch.device("cpu"),
        build_model_fn=lambda *args: pytest.fail("model should not be built"),
    )

    assert written["path"] == "predictions.arrow"
    assert written["col"] == "out"
    assert written["preds"].shape == (0, 6)
    assert written["preds"].dtype == torch.float32


def test_stream_predict_rejects_checkpoint_feature_dim_mismatch(monkeypatch):
    args = model_args(
        seq_len=None,
        hidden=None,
        layers=None,
        dropout=None,
        nhead=None,
        context_mode=None,
        out_dim=None,
        pred_col="predictions",
    )
    table = SimpleNamespace(num_rows=1)

    monkeypatch.setattr(
        predict_stream_module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    monkeypatch.setattr(
        predict_stream_module,
        "iter_framed_arrow",
        lambda *args, **kwargs: iter([table]),
    )
    monkeypatch.setattr(
        predict_stream_module,
        "load_checkpoint_metadata",
        lambda *args: {"model_config": checkpoint_model_config(feature_dim=2)},
    )
    monkeypatch.setattr(
        predict_stream_module,
        "table_to_source_tensor",
        lambda _: torch.randn(1, 6),
    )

    with pytest.raises(
        ValueError,
        match="Feature dim 3 does not match checkpoint feature_dim 2",
    ):
        predict_stream_module.run(args, torch.device("cpu"))


def test_stream_predict_keeps_cross_frame_check_for_legacy_checkpoint(monkeypatch):
    args = model_args(
        seq_len=None,
        hidden=None,
        layers=None,
        dropout=None,
        nhead=None,
        context_mode=None,
        out_dim=None,
        pred_col="predictions",
    )
    first = SimpleNamespace(num_rows=1, frame=1)
    second = SimpleNamespace(num_rows=1, frame=2)

    monkeypatch.setattr(
        predict_stream_module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    monkeypatch.setattr(
        predict_stream_module,
        "iter_framed_arrow",
        lambda *args, **kwargs: iter([first, second]),
    )
    monkeypatch.setattr(
        predict_stream_module,
        "load_checkpoint_metadata",
        lambda *args: {"model_config": checkpoint_model_config(feature_dim=None)},
    )
    monkeypatch.setattr(
        predict_stream_module,
        "table_to_source_tensor",
        lambda table: torch.randn(1, 4 if table.frame == 1 else 6),
    )
    monkeypatch.setattr(predict_stream_module, "write_framed_arrow", lambda *args: None)

    class Trainer:
        def load(self, *args):
            pass

        def predict(self, X):
            return torch.zeros((X.shape[0], 6))

    with pytest.raises(
        ValueError,
        match="Feature dim changed across frames: expected 2, got 3",
    ):
        predict_stream_module.run(
            args,
            torch.device("cpu"),
            build_model_fn=lambda *args: object(),
            build_trainer_fn=lambda *args: Trainer(),
        )
