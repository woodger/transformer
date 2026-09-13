import sys
from types import SimpleNamespace

import pytest

from app import __version__
from app.cli.args import parse_args
from app.version import __version__ as runtime_version, version_text
from app.worker.application.capabilities import inspect_capabilities
from app.worker.runtime.device import get_device


def test_cpu_device():
    device = get_device("cpu")
    assert device.type == "cpu"


def test_worker_capabilities_follow_the_v13_contract(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.12.0+test",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    document = inspect_capabilities()

    assert document == {
        "contract": "transformer-worker",
        "protocolVersion": 13,
        "checkpointFormat": "transformer-checkpoint-v7",
        "recoveryFormat": "transformer-recovery-v7",
        "schemaIds": {
            "fitInput": "transformer.indexed-feature-blocks.fit.v1",
            "predictInput": "transformer.indexed-feature-blocks.predict.v1",
            "predictionOutput": "transformer.prediction.target-aligned.v3",
        },
        "semantic": {
            "objectiveLanguage": {
                "revision": 2,
                "constraints": ["ClosedInterval", "Finite"],
                "transformations": ["Identity", "Sigmoid", "Tanh"],
                "resourceClasses": ["PositiveScalarPerObservation"],
                "directOperators": [
                    "BinaryCrossEntropyWithLogits",
                    "LogMSE",
                    "SmoothL1",
                ],
                "auxiliaryOperators": [
                    "ExpectedValue",
                    "GaussianNLL",
                    "RiskAdjustedExpectedValue",
                ],
                "aggregations": ["WeightedSum"],
                "reductions": ["GlobalRowMean"],
            },
            "semanticLimits": {
                "maxTargetSlots": 128,
                "maxObjectiveComponents": 256,
                "maxPrivateResources": 64,
            },
            "modelArchitectures": [{
                "identity": "transformer.sequence-model",
                "revision": 1,
            }],
        },
        "torchVersion": "2.12.0+test",
        "cudaRuntimeVersion": "13.0",
        "devices": [{
            "backend": "cpu",
            "opaqueId": "cpu",
            "name": "CPU",
        }],
    }


def test_gpu_device_alias_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device: gpu"):
        get_device("gpu")


def test_package_reexports_runtime_version():
    assert __version__ == runtime_version


@pytest.mark.parametrize("option", ("--version", "-v"))
def test_cli_version(capsys, monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["transformer", option])

    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert output == f"{version_text('transformer')}\n"


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
  --version, -v    Show package version and exit

Commands:

Flight:
  flight serve                    Run the durable Arrow Flight job service.

Access:
  auth tokens issue               Issue a local API access token
  auth tokens list                List API access token metadata
  auth tokens revoke <token-id>   Revoke an API access token

Models:
  models list                     List published model generations
  models delete <model-ref>       Delete one model generation

Training and inference:
  fit                             Train from an Arrow file.
  predict                         Predict from an Arrow file.
  fit-stream                      Train from framed stdin.
  predict-stream                  Predict from framed stdin.

Diagnostics:
  gmark                           Stress one CUDA GPU with synthetic training.

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
    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "fit-stream",
        "--model-contract=contract.json",
        "--use-amp",
    ])

    args = parse_args()

    assert args.use_amp is True


def test_cli_rejects_removed_loss_schedule_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "fit-stream",
        "--seq-len=12",
        "--loss-stage=4",
        "--loss-schedule=step",
        "--stage-size=100",
    ])

    with pytest.raises(SystemExit):
        parse_args()
