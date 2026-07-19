import io
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

import app.main as main_module
from app.metrics import TrainMetrics


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
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 1,
        "patience": 1,
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

        def fit_epochs(self, X, Y, on_epoch=None, frame=None):
            self.calls.append((X.shape, Y.shape))
            metrics_rows = [
                TrainMetrics(rows=1, batches=1, loss=1.25, loss_stage=1),
                TrainMetrics(rows=1, batches=1, loss=1.10, loss_stage=2),
            ]
            for epoch, metrics in enumerate(metrics_rows):
                on_epoch(epoch, metrics, {
                    "monitor_value": metrics.loss,
                    "baseline_passed": True,
                    "checkpoint_best": epoch == 0,
                    "best_monitor": metrics.loss,
                })
            return metrics_rows

        def save(self, model_name):
            self.saved_as = model_name

        def record_metrics(self, metrics, **extra):
            pass

    trainer = FakeTrainer()

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(stream))
    monkeypatch.setattr(main_module, "build_model", lambda *args: object())
    monkeypatch.setattr(main_module, "build_trainer", lambda *args: trainer)

    main_module.fit_stream(make_args(model_name="stream.pth"), torch.device("cpu"))

    assert trainer.calls == [(torch.Size([1, 2, 2]), torch.Size([1, 6]))]
    assert trainer.saved_as == "stream.pth"

    output = capsys.readouterr().out
    assert "frame 1, skipped empty payload" in output
    assert "frame=2 epoch=1 monitor_value=1.25" in output
    assert "loss=1.250000" in output
    assert "frame=2 epoch=2 monitor_value=1.1" in output
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

    with pytest.raises(ValueError, match="No non-empty frames received"):
        main_module.fit_stream(make_args(), torch.device("cpu"))


def test_fit_stream_spool_runs_epochs_over_all_payloads(tmp_path, monkeypatch, capsys):
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
        "tgt": pa.array([], type=pa.list_(pa.float32())),
    })
    first = pa.table({
        "src": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        "tgt": [
            [0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.4, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
    })
    second = pa.table({
        "src": [[9.0, 10.0, 11.0, 12.0]],
        "tgt": [[0.3, 0.0, 0.0, 0.0, 1.0, 1.0]],
    })
    spool = tmp_path / "inputs"
    spool.mkdir()
    for ordinal, table in enumerate((empty, first, second)):
        write_arrow_table(spool / f"{ordinal}.arrow", table)

    class FakeTrainer:
        def __init__(self):
            self.payload_passes = []
            self.metrics_rows = []
            self.saved_as = None

        def fit_payloads(self, payloads, on_epoch=None):
            for epoch in range(2):
                loaded = list(payloads())
                self.payload_passes.append([
                    (X.shape, Y.shape)
                    for X, Y in loaded
                ])
                metrics = TrainMetrics(
                    rows=sum(X.size(0) for X, _ in loaded),
                    batches=len(loaded),
                    loss=1.0 - epoch * 0.1,
                    loss_stage=epoch + 1,
                )
                on_epoch(epoch, metrics, {
                    "monitor_value": metrics.loss,
                    "baseline_passed": True,
                    "checkpoint_best": True,
                    "best_monitor": metrics.loss,
                })

        def save(self, model_name):
            self.saved_as = model_name

        def record_metrics(self, metrics, **extra):
            self.metrics_rows.append((metrics, extra))

    trainer = FakeTrainer()
    monkeypatch.setattr(main_module, "build_model", lambda *args: object())
    monkeypatch.setattr(main_module, "build_trainer", lambda *args: trainer)

    main_module.fit_stream(
        make_args(
            model_name="spooled.pth",
            input_spool_dir=str(spool),
            input_frame_count=3,
        ),
        torch.device("cpu"),
    )

    expected_pass = [
        (torch.Size([2, 2, 2]), torch.Size([2, 6])),
        (torch.Size([1, 2, 2]), torch.Size([1, 6])),
    ]
    assert trainer.payload_passes == [expected_pass, expected_pass]
    assert trainer.saved_as == "spooled.pth"
    assert [extra["epoch"] for _, extra in trainer.metrics_rows] == [1, 2]
    assert all("frame" not in extra for _, extra in trainer.metrics_rows)

    output = capsys.readouterr().out
    assert "frame 1, skipped empty payload" in output
    assert "epoch=1 monitor_value=1" in output
    assert "frame=" not in output.split("epoch=1", 1)[1]
    assert "2 trained frame(s), 2 epoch(s) from 3 received frame(s)" in output


def test_fit_stream_applies_max_frame_bytes(monkeypatch):
    input_stream = io.BytesIO((11).to_bytes(8, byteorder="big") + b"payload")
    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(input_stream))

    with pytest.raises(ValueError, match="exceeds maximum 10 bytes"):
        main_module.fit_stream(
            make_args(max_frame_bytes=10),
            torch.device("cpu"),
        )


def test_main_rejects_data_path_for_fit_stream(monkeypatch):
    args = make_args(action="fit-stream", data="train.arrow", device="cpu")
    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(main_module, "get_device", lambda device: torch.device("cpu"))

    with pytest.raises(ValueError, match="data path is not supported"):
        main_module.main()
