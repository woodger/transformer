import sys

import pytest
from app.runtime.device import get_device
from app import __version__
from app.cli.args import parse_args
from app.cli.help import build_parser
from app.runtime.version import __version__ as runtime_version
from app.runtime.version import version_text


def test_cpu_device():
    device = get_device("cpu")
    assert device.type == "cpu"


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    ((True, "cuda"), (False, "cpu")),
)
def test_legacy_gpu_device_follows_cuda_availability(
    monkeypatch,
    cuda_available,
    expected,
):
    monkeypatch.setattr(
        "app.runtime.device.torch.cuda.is_available",
        lambda: cuda_available,
    )

    assert get_device("gpu").type == expected


def test_package_reexports_runtime_version():
    assert __version__ == runtime_version


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

Flight:
  flight serve                    Run the durable Arrow Flight job service.

Access:
  auth tokens issue               Issue a local API access token
  auth tokens list                List API access token metadata
  auth tokens revoke <token-id>   Revoke an API access token

Training and inference:
  fit                             Train from an Arrow file.
  predict                         Predict from an Arrow file.
  fit-stream                      Train from framed stdin.
  predict-stream                  Predict from framed stdin.

Metrics:
  plot-metrics                    Render SVG charts from metrics JSONL.

Database:
  db migrations status            Read-only schema migration state
  db migrations apply             Apply pending schema migrations
  db migrations rollback          Revert the latest schema migration

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
