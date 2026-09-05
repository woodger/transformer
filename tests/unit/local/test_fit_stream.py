import io
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

import app.main as main_module
from app.worker.data.tensors import TrainingBatch
from app.worker.telemetry import ObservedTrainingEpoch
from tests.support.consumer_neutral import model_contract


class FakeStdin:
    def __init__(self, buffer):
        self.buffer = buffer


def write_framed_table(stream, table):
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

    payload = sink.getvalue().to_pybytes()
    stream.write(len(payload).to_bytes(8, byteorder="big", signed=False))
    stream.write(payload)


def write_arrow_table(path, table):
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def make_args(**overrides):
    args = {
        "seq_len": 2,
        "model_name": "model.pth",
        "model_contract": "model-contract.json",
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 1,
        "use_amp": False,
        "hidden": 32,
        "layers": 1,
        "dropout": 0.0,
        "nhead": 4,
        "seed": 42,
        "deterministic": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_fit_stream_skips_empty_frames(monkeypatch, capsys):
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
        "tgt": pa.array([], type=pa.list_(pa.float32())),
    })
    non_empty = pa.table({
        "src": [[1.0, 2.0, 3.0, 4.0]],
        "tgt": [[0.5]],
    })

    stream = io.BytesIO()
    write_framed_table(stream, empty)
    write_framed_table(stream, non_empty)
    stream.seek(0)

    class FakeTrainer:
        def __init__(self):
            self.calls = []
            self.saved_as = None

        def fit_epochs(
            self,
            batch: TrainingBatch,
            on_epoch=None,
            frame=None,
        ):
            self.calls.append((batch.features.shape, batch.targets.shape))
            metrics_rows = [
                ObservedTrainingEpoch(
                    targets=("MeanReturn",),
                    direct_components=(("direct.mean-return", "SmoothL1"),),
                    rows=1,
                    batches=1,
                    loss=1.25,
                ),
                ObservedTrainingEpoch(
                    targets=("MeanReturn",),
                    direct_components=(("direct.mean-return", "SmoothL1"),),
                    rows=1,
                    batches=1,
                    loss=1.10,
                ),
            ]
            for epoch, metrics in enumerate(metrics_rows):
                on_epoch(epoch, metrics, {
                    "selection_score": None,
                    "checkpoint_best": epoch == 0,
                    "should_stop": False,
                    "best_selection_score": metrics.loss,
                })
            return metrics_rows

        def config_line(self):
            return "test trainer"

        def save(self, model_name, *, metadata):
            self.saved_as = model_name

        def record_metrics(self, metrics, **extra):
            pass

    trainer = FakeTrainer()
    contract = model_contract(
        "single-regression",
        seq_len=2,
        feature_dim=2,
        hidden=32,
        layers=1,
        dropout=0.0,
        nhead=4,
    )

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(stream))
    monkeypatch.setattr(
        "app.local.fit_stream.load_model_contract",
        lambda _path: contract,
    )
    monkeypatch.setattr(
        "app.local.fit_stream.local_checkpoint_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(main_module, "build_model", lambda *args: object())
    trainer_arguments = {}

    def build_trainer(*args, **kwargs):
        trainer_arguments.update(kwargs)
        return trainer

    monkeypatch.setattr(
        "app.worker.training.factory.build_trainer",
        build_trainer,
    )

    main_module.fit_stream(make_args(model_name="stream.pth"), torch.device("cpu"))

    assert trainer.calls == [(torch.Size([1, 2, 2]), torch.Size([1, 1]))]
    assert trainer.saved_as == "stream.pth"
    assert trainer_arguments["model_contract"] is contract
    assert trainer_arguments["initialization"] == {"kind": "random"}

    output = capsys.readouterr().out
    assert "frame 1, skipped empty payload" in output
    assert "features: torch.Size([1, 2, 2]) targets: torch.Size([1, 1])" in output
    assert "frame=2 epoch=1 selection=n/a" in output
    assert "loss=1.250000" in output
    assert "frame=2 epoch=2 selection=n/a" in output
    assert "loss=1.100000" in output
    assert "Model saved after 1 trained frame(s), 2 epoch(s) from 2 received frame(s)" in output


def test_fit_stream_rejects_all_empty_frames(monkeypatch):
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
        "tgt": pa.array([], type=pa.list_(pa.float32())),
    })

    stream = io.BytesIO()
    write_framed_table(stream, empty)
    stream.seek(0)

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(stream))
    monkeypatch.setattr(
        "app.local.fit_stream.load_model_contract",
        lambda _path: model_contract(
            "single-regression",
            seq_len=2,
            feature_dim=2,
            hidden=32,
            layers=1,
            dropout=0.0,
            nhead=4,
        ),
    )

    with pytest.raises(ValueError, match="No non-empty frames received"):
        main_module.fit_stream(make_args(), torch.device("cpu"))


def test_fit_stream_applies_max_frame_bytes(monkeypatch):
    input_stream = io.BytesIO((11).to_bytes(8, byteorder="big") + b"payload")
    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(input_stream))
    monkeypatch.setattr(
        "app.local.fit_stream.load_model_contract",
        lambda _path: model_contract(
            "single-regression",
            seq_len=2,
            feature_dim=2,
            hidden=32,
            layers=1,
            dropout=0.0,
            nhead=4,
        ),
    )

    with pytest.raises(ValueError, match="exceeds maximum 10 bytes"):
        main_module.fit_stream(
            make_args(max_frame_bytes=10),
            torch.device("cpu"),
        )


def test_main_rejects_data_path_for_fit_stream(monkeypatch):
    args = make_args(action="fit-stream", data="train.arrow", device="cpu")
    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        main_module,
        "configure_reproducibility",
        lambda seed, deterministic: None,
    )
    monkeypatch.setattr(main_module, "get_device", lambda device: torch.device("cpu"))

    with pytest.raises(ValueError, match="data path is not supported"):
        main_module.main()
