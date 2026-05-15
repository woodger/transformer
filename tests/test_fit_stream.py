import io
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

import app.main as main_module


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


def make_args(**overrides):
    args = {
        "seq_len": 2,
        "model_name": "model.pth",
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


def test_fit_stream_skips_empty_frames(monkeypatch, capsys):
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
        "tgt": pa.array([], type=pa.list_(pa.float32())),
    })
    non_empty = pa.table({
        "src": [[1.0, 2.0, 3.0, 4.0]],
        "tgt": [[0.5, 0.0, 0.0, 0.0, 1.0, 1.0]],
    })

    stream = io.BytesIO()
    write_framed_table(stream, empty)
    write_framed_table(stream, non_empty)
    stream.seek(0)

    class FakeTrainer:
        def __init__(self):
            self.calls = []
            self.saved_as = None

        def fit_batch(self, X, Y, epoch):
            self.calls.append((X.shape, Y.shape, epoch))
            return 1.25

        def save(self, model_name):
            self.saved_as = model_name

    trainer = FakeTrainer()

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(stream))
    monkeypatch.setattr(main_module, "build_model", lambda *args: object())
    monkeypatch.setattr(main_module, "build_trainer", lambda *args: trainer)

    main_module.fit_stream(make_args(model_name="stream.pth"), torch.device("cpu"))

    assert trainer.calls == [(torch.Size([1, 2, 2]), torch.Size([1, 6]), 0)]
    assert trainer.saved_as == "stream.pth"

    output = capsys.readouterr().out
    assert "frame 1, skipped empty payload" in output
    assert "frame 2, loss 1.250000" in output
    assert "Model saved after 1 trained frame(s) from 2 received frame(s)" in output


def test_fit_stream_rejects_all_empty_frames(monkeypatch):
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
        "tgt": pa.array([], type=pa.list_(pa.float32())),
    })

    stream = io.BytesIO()
    write_framed_table(stream, empty)
    stream.seek(0)

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(stream))

    with pytest.raises(ValueError, match="No non-empty frames received"):
        main_module.fit_stream(make_args(), torch.device("cpu"))


def test_main_rejects_data_path_for_fit_stream(monkeypatch):
    args = make_args(action="fit-stream", data="train.arrow", device="cpu")
    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(main_module, "get_device", lambda device: torch.device("cpu"))

    with pytest.raises(ValueError, match="data path is not supported"):
        main_module.main()
