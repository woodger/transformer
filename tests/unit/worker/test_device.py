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


def test_worker_capabilities_follow_the_v21_contract(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.12.0+test",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    document = inspect_capabilities()

    assert document == {
        "contract": "transformer-worker",
        "protocolVersion": 21,
        "checkpointFormat": "transformer-checkpoint-v13",
        "recoveryFormat": "transformer-recovery-v13",
        "schemaIds": {
            "fitInput": "transformer.indexed-feature-blocks.fit.v1",
            "predictInput": "transformer.indexed-feature-blocks.predict.v1",
            "predictionOutput": "transformer.prediction.target-aligned.v3",
        },
        "semantic": {
            "objectiveLanguage": {
                "revision": 6,
                "closed": True,
            },
            "semanticLimits": {
                "maxTargetSlots": 128,
                "maxObjectiveComponents": 256,
                "maxPrivateResources": 64,
            },
            "directOperators": [
                "BinaryCrossEntropyWithLogits",
                "LogMSE",
                "PositiveClassWeightedBinaryCrossEntropyWithLogits",
                "SmoothL1",
            ],
            "auxiliaryOperators": [
                "BernoulliConfidencePenalty",
                "ExpectedValue",
                "GaussianNLL",
                "RiskAdjustedExpectedValue",
            ],
            "encoderNormalizationOrders": ["postNorm", "preNorm"],
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


def test_cli_help_exposes_only_current_service_commands(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["transformer", "--help"])

    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert output.startswith(f"transformer {__version__}\n")
    assert "flight serve" in output
    assert "fit-stream" not in output
    assert "predict-stream" not in output
