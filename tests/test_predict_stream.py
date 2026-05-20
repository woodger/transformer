import io
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

import app.main as main_module
from app.arrow_io import iter_framed_arrow


class FakeStdin:
    def __init__(self, buffer):
        self.buffer = buffer


class FakeStdout:
    def __init__(self, buffer):
        self.buffer = buffer


def write_framed_table(stream, table):
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

    payload = sink.getvalue().to_pybytes()
    stream.write(len(payload).to_bytes(8, byteorder="big", signed=False))
    stream.write(payload)


def make_args(**overrides):
    args = {
        "seq_len": 2,
        "model_name": "model.pth",
        "pred_col": "out",
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 1,
        "patience": 1,
        "use_amp": False,
        "hidden": 32,
        "layers": 1,
        "dropout": 0.0,
        "nhead": 4,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_predict_stream_writes_framed_predictions(monkeypatch):
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
    })
    first = pa.table({"src": [[1.0, 2.0, 3.0, 4.0]]})
    second = pa.table({"src": [[5.0, 6.0, 7.0, 8.0]]})

    input_stream = io.BytesIO()
    write_framed_table(input_stream, empty)
    write_framed_table(input_stream, first)
    write_framed_table(input_stream, second)
    input_stream.seek(0)

    output_stream = io.BytesIO()

    class FakeTrainer:
        def __init__(self):
            self.loaded = []
            self.calls = []

        def load(self, model_name):
            self.loaded.append(model_name)

        def predict(self, X):
            self.calls.append(X.shape)
            value = float(len(self.calls))
            return torch.tensor([[value, value + 1.0]], dtype=torch.float32)

    trainer = FakeTrainer()

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(input_stream))
    monkeypatch.setattr(main_module.sys, "stdout", FakeStdout(output_stream))
    monkeypatch.setattr(main_module, "build_model", lambda *args: object())
    monkeypatch.setattr(main_module, "build_trainer", lambda *args: trainer)

    main_module.predict_stream(make_args(), torch.device("cpu"))

    assert trainer.loaded == ["model.pth"]
    assert trainer.calls == [torch.Size([1, 2, 2]), torch.Size([1, 2, 2])]

    output_stream.seek(0)
    frames = list(iter_framed_arrow(output_stream))

    assert len(frames) == 3
    assert frames[0].column("out").to_pylist() == []
    assert frames[1].column("out").to_pylist() == [[1.0, 2.0]]
    assert frames[2].column("out").to_pylist() == [[2.0, 3.0]]


def test_main_rejects_data_path_for_predict_stream(monkeypatch):
    args = make_args(
        action="predict-stream",
        data="predict.arrow",
        device="cpu",
    )
    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(main_module, "get_device", lambda device: torch.device("cpu"))

    with pytest.raises(ValueError, match="data path is not supported"):
        main_module.main()
