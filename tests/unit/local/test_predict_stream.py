import importlib
import io
import uuid
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

import app.main as main_module
import app.worker.checkpoints.model as checkpoint_module
from app.contracts.worker.v12.config import TrainConfig, train_config_to_manifest
from app.worker.data.arrow import iter_framed_arrow
from tests.support.consumer_neutral import data_contract, model_contract

predict_stream_module = importlib.import_module("app.local.predict_stream")
MODEL_CONTRACT_VALUE = model_contract(
    "single-regression",
    seq_len=2,
    feature_dim=2,
    hidden=32,
    layers=1,
    dropout=0.0,
    nhead=4,
)


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
        "use_amp": False,
        "hidden": 32,
        "layers": 1,
        "dropout": 0.0,
        "nhead": 4,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def checkpoint_payload():
    data = data_contract(MODEL_CONTRACT_VALUE)

    return {
        "metadata": {
            "dataContract": data,
            "modelContract": MODEL_CONTRACT_VALUE.to_document(),
            "semanticDigests": MODEL_CONTRACT_VALUE.digests(
                data["dataContractSha256"],
            ),
        },
        "state_dict": {},
    }


def checkpoint_metadata():
    contract = MODEL_CONTRACT_VALUE
    data = data_contract(contract)
    digests = contract.digests(data["dataContractSha256"])
    training = TrainConfig()
    return {
        "format": "transformer-checkpoint-v6",
        "serviceVersion": "test",
        "generation": 1,
        "jobId": str(uuid.uuid4()),
        "dataContract": data,
        "modelContract": contract.to_document(),
        "semanticDigests": digests,
        "trainingConfig": train_config_to_manifest(training),
        "diagnostics": training.diagnostics.to_document(),
        "selection": {
            "enabled": False,
            "modelContractSha256": digests["modelContractSha256"],
            "bestSelectionScore": None,
            "bestEpoch": None,
            "source": "last_epoch",
        },
        "initialization": {"kind": "random"},
        "jobConfigSha256": "a" * 64,
        "manifestSha256": "b" * 64,
        "progress": {
            "completedEpochs": 1,
            "globalStep": 1,
            "trainingComplete": True,
        },
    }


def test_predict_stream_writes_framed_predictions(monkeypatch, capsys):
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

        def load_payload(self, checkpoint):
            print("accidental load stdout")
            self.loaded.append(checkpoint)

        def predict(self, features):
            print("accidental predict stdout")
            self.calls.append(features.shape)
            value = float(len(self.calls)) / 10.0
            return torch.tensor(
                [[value]],
                dtype=torch.float32,
            )

    trainer = FakeTrainer()
    checkpoint = checkpoint_payload()
    checkpoint_loads = []

    def load_checkpoint_once(*args):
        checkpoint_loads.append(args)
        return checkpoint

    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(input_stream))
    monkeypatch.setattr(main_module.sys, "stdout", FakeStdout(output_stream))
    monkeypatch.setattr(
        predict_stream_module,
        "load_checkpoint",
        load_checkpoint_once,
    )

    def build_model(*args):
        print("accidental build stdout")
        return object()

    monkeypatch.setattr(main_module, "build_model", build_model)
    monkeypatch.setattr(
        main_module,
        "build_trainer",
        lambda *args, **kwargs: trainer,
    )

    main_module.predict_stream(make_args(), torch.device("cpu"))

    assert trainer.loaded == [checkpoint]
    assert len(checkpoint_loads) == 1
    assert trainer.calls == [torch.Size([1, 2, 2]), torch.Size([1, 2, 2])]

    output_stream.seek(0)
    frames = list(iter_framed_arrow(output_stream))

    assert len(frames) == 3
    assert frames[0].column("out").to_pylist() == []
    assert frames[1].column("out").to_pylist() == [
        pytest.approx([0.1])
    ]
    assert frames[2].column("out").to_pylist() == [
        pytest.approx([0.2])
    ]
    diagnostics = capsys.readouterr().err
    assert "accidental build stdout" in diagnostics
    assert "accidental load stdout" in diagnostics
    assert "accidental predict stdout" in diagnostics
    assert "features: torch.Size([1, 2, 2])" in diagnostics


def test_predict_stream_applies_max_frame_bytes_without_writing_stdout(monkeypatch):
    input_stream = io.BytesIO((11).to_bytes(8, byteorder="big") + b"payload")
    output_stream = io.BytesIO()
    monkeypatch.setattr(main_module.sys, "stdin", FakeStdin(input_stream))
    monkeypatch.setattr(main_module.sys, "stdout", FakeStdout(output_stream))
    monkeypatch.setattr(
        predict_stream_module,
        "load_checkpoint",
        lambda *args: checkpoint_payload(),
    )

    with pytest.raises(ValueError, match="exceeds maximum 10 bytes"):
        main_module.predict_stream(
            make_args(max_frame_bytes=10),
            torch.device("cpu"),
        )

    assert output_stream.getvalue() == b""


def test_predict_stream_loads_checkpoint_once_for_all_empty_input(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_module.save_checkpoint(
        checkpoint_path,
        torch.nn.Linear(1, 1),
        metadata=checkpoint_metadata(),
    )
    empty = pa.table({
        "src": pa.array([], type=pa.list_(pa.float32())),
    })
    input_stream = io.BytesIO()
    write_framed_table(input_stream, empty)
    input_stream.seek(0)
    output_stream = io.BytesIO()
    original_torch_load = checkpoint_module.torch.load
    calls = []

    def counting_torch_load(*args, **kwargs):
        calls.append((args, kwargs))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", counting_torch_load)
    monkeypatch.setattr(
        predict_stream_module.sys,
        "stdin",
        FakeStdin(input_stream),
    )
    monkeypatch.setattr(
        predict_stream_module.sys,
        "stdout",
        FakeStdout(output_stream),
    )

    predict_stream_module.run(
        make_args(model_name=str(checkpoint_path)),
        torch.device("cpu"),
        build_model_fn=lambda *args: pytest.fail("model should not be built"),
    )

    assert len(calls) == 1
    output_stream.seek(0)
    frames = list(iter_framed_arrow(output_stream))
    assert len(frames) == 1
    field = frames[0].schema.field("out")
    assert field.type == pa.list_(pa.float32(), 1)
    assert field.nullable is False
    assert frames[0].num_rows == 0


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
