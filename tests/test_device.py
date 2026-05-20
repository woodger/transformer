import sys

import torch

from app.device import get_device
from app import __version__
from app.args import parse_args
from app.help import build_parser


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
    assert __version__ == "0.1.0"


def test_cli_version(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--version"])

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out.strip()
    assert output == "main.py 0.1.0"


def test_cli_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "Transformer training and inference CLI." in output
    assert "Modes:" in output
    assert "Runtime:" in output
    assert "Training:" in output
    assert "Examples:" in output
    assert "--metrics-name" in output


def test_parser_is_buildable():
    parser = build_parser()
    assert parser.prog == "main.py"
