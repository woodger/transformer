import sys

import pytest
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


@pytest.mark.parametrize("option", ("--version", "-v"))
def test_cli_version(capsys, monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["transformer", option])

    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert version_text("transformer") in output
    assert "python " in output
    assert "torch " in output
    assert "cuda " in output


def test_cli_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["transformer", "--help"])

    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert output == f"""transformer {__version__}

Usage:
  transformer <command> [args] [options]
  transformer <command> --help
  transformer --help
  transformer --version

Global options:
  --help, -h       Show help and exit
  --version, -v    Show package and runtime version info

Commands:
  fit             Train from an Arrow file.
  predict         Predict from an Arrow file.
  fit-stream      Train from framed stdin.
  predict-stream  Predict from framed stdin.
  serve-flight    Run the durable Arrow Flight job service.
  plot-metrics    Render SVG charts from metrics JSONL.

Command details:
  transformer <command> --help
"""


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
    assert parser.prog == "transformer"
