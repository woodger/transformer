import sys

import torch

from app.runtime.device import get_device
from app import __version__
from app.cli.args import parse_args
from app.cli.help import build_parser
from app.runtime.version import version_text


def test_cpu_device():
    device = get_device("cpu")
    assert device.type == "cpu"


def test_gpu_fallback_to_cpu():
    device = get_device("gpu")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_version_is_exported():
    assert __version__


def test_cli_version(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--version"])

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert version_text("main.py") in output
    assert "python " in output
    assert "torch " in output
    assert "cuda " in output


def test_cli_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "Transformer training and inference CLI." in output
    assert "Data contract:" in output
    assert "Modes:" in output
    assert "Streaming protocol:" in output
    assert "Storage:" in output
    assert "Context modes:" in output
    assert "Runtime:" in output
    assert "Training:" in output
    assert "Examples:" in output
    assert "--metrics-name" in output
    assert "--mode" in output
    assert "--loss-stage" in output
    assert "--loss-schedule" in output
    assert "--stage-size" in output
    assert "--per-week" not in output
    assert "--context-mode" not in output
    assert "--use-amp" in output
    assert "--amp" not in output


def test_cli_use_amp(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "fit-stream", "--seq-len=12", "--use-amp"])

    args = parse_args()

    assert args.use_amp is True


def test_cli_loss_schedule_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "fit-stream",
        "--seq-len=12",
        "--loss-stage=3",
        "--loss-schedule=step",
        "--stage-size=100",
    ])

    args = parse_args()

    assert args.loss_stage == 3
    assert args.loss_schedule == "step"
    assert args.stage_size == 100


def test_parser_is_buildable():
    parser = build_parser()
    assert parser.prog == "main.py"
