from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import app.commands.gmark as gmark
import app.main as main_module
from app.cli.help import build_parser


def _args(**overrides):
    values = {
        "duration": 0.01,
        "device": 0,
        "matrix_size": 64,
        "dtype": "float16",
        "memory_fraction": 0.0,
        "max_temperature": 0.0,
        "status_interval": 0.005,
        "warmup_iterations": 1,
        "sync_every": 1,
        "seed": 12345,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gmark_reports_unavailable_cuda_without_starting_a_workload(capsys):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda="13.0"),
    )

    result = gmark.run(_args(), torch_module=fake_torch)

    assert result == 1
    assert capsys.readouterr().err == (
        "ERROR: CUDA is not available to PyTorch "
        "(PyTorch CUDA build: 13.0)\n"
    )


def test_gmark_rejects_a_nonexistent_logical_device(capsys):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        ),
        version=SimpleNamespace(cuda="13.0"),
    )

    result = gmark.run(_args(device=1), torch_module=fake_torch)

    assert result == 1
    assert "CUDA device 1 does not exist; visible device count is 1" in (
        capsys.readouterr().err
    )


def test_temperature_cutoff_fails_closed_before_allocating_matrices(
    monkeypatch,
    capsys,
):
    class FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def set_device(self, _index):
            pass

        def get_device_properties(self, _device):
            return SimpleNamespace(
                name="Test GPU",
                major=9,
                minor=0,
                uuid="GPU-test",
            )

        def is_bf16_supported(self):
            return True

        def manual_seed_all(self, _seed):
            pass

        def device(self, _index):
            return nullcontext()

        def mem_get_info(self):
            return 8 * gmark.GIB, 8 * gmark.GIB

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
        device=lambda *_args: object(),
        float16=object(),
        bfloat16=object(),
        float32=object(),
        manual_seed=lambda _seed: None,
    )

    class HotMonitor:
        def __init__(self, _identifiers):
            pass

        def read(self):
            return gmark.GpuMetrics(85.0, None, 0.0, 20.0)

    monkeypatch.setattr(gmark, "NvidiaSmiMonitor", HotMonitor)

    result = gmark.run(
        _args(max_temperature=80.0),
        torch_module=fake_torch,
    )

    assert result == 1
    output = capsys.readouterr()
    assert "GPU: Test GPU (cuda:0)" in output.out
    assert "GPU is already at 85 C" in output.err


def test_temperature_cutoff_stops_an_active_workload_with_exit_three(
    monkeypatch,
    capsys,
):
    class FakeTensor:
        def __getitem__(self, _index):
            return self

        def clone(self):
            return self

    class FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def set_device(self, _index):
            pass

        def get_device_properties(self, _device):
            return SimpleNamespace(
                name="Test GPU",
                major=9,
                minor=0,
                uuid="GPU-test",
            )

        def is_bf16_supported(self):
            return True

        def manual_seed_all(self, _seed):
            pass

        def device(self, _index):
            return nullcontext()

        def mem_get_info(self):
            return 8 * gmark.GIB, 8 * gmark.GIB

        def synchronize(self, _device):
            pass

        def reset_peak_memory_stats(self, _device):
            pass

        def max_memory_allocated(self, _device):
            return 1024

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
        device=lambda *_args: object(),
        float16=object(),
        bfloat16=object(),
        float32=object(),
        manual_seed=lambda _seed: None,
        randn=lambda *_args, **_kwargs: FakeTensor(),
        empty_like=lambda _tensor: FakeTensor(),
        mm=lambda *_args, **_kwargs: None,
        isfinite=lambda _tensor: SimpleNamespace(
            all=lambda: SimpleNamespace(item=lambda: True)
        ),
        allclose=lambda *_args, **_kwargs: True,
        OutOfMemoryError=RuntimeError,
    )
    readings = iter((
        gmark.GpuMetrics(60.0, None, 0.0, 20.0),
        gmark.GpuMetrics(60.0, None, 0.0, 20.0),
        gmark.GpuMetrics(85.0, None, 100.0, 250.0),
    ))

    class HeatingMonitor:
        def __init__(self, _identifiers):
            pass

        def read(self):
            return next(readings)

    clock = iter((0.0, 0.0, 0.2, 0.2))
    monkeypatch.setattr(gmark, "NvidiaSmiMonitor", HeatingMonitor)
    monkeypatch.setattr(gmark.time, "monotonic", lambda: next(clock))

    result = gmark.run(
        _args(
            duration=1.0,
            status_interval=0.1,
            max_temperature=80.0,
        ),
        torch_module=fake_torch,
    )

    assert result == 3
    output = capsys.readouterr()
    assert "Temperature cutoff reached: 85 C >= 80 C." in output.err
    assert "FAIL: stopped by the temperature safety cutoff." in output.out


def test_nvidia_smi_monitor_accepts_devices_without_a_vram_sensor(
    monkeypatch,
):
    commands = []

    def query(command, **kwargs):
        commands.append((command, kwargs))
        if "temperature.memory" in command[2]:
            return SimpleNamespace(stdout="not supported\n")
        return SimpleNamespace(stdout="67, 98, 245.5\n")

    monkeypatch.setattr(gmark.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gmark.subprocess, "run", query)

    metrics = gmark.NvidiaSmiMonitor(("GPU-id", "0")).read()

    assert metrics == gmark.GpuMetrics(
        temperature_c=67.0,
        memory_temperature_c=None,
        utilization_percent=98.0,
        power_w=245.5,
    )
    assert len(commands) == 2
    assert commands[0][0][0] == "/usr/bin/nvidia-smi"
    assert commands[0][0][1] == "--id=GPU-id"
    assert commands[0][1] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 2,
    }


def test_periodic_status_omits_vram_usage_and_capacity():
    status = gmark._format_status(
        2.0,
        12,
        3.5,
        gmark.GpuMetrics(67.0, 72.0, 98.0, 245.5),
    )

    assert status == (
        "[    2.0s] | iterations=12 | average=3.50 TFLOP/s | "
        "gpu-temp=67 C | vram-temp=72 C | util=98% | power=245.5 W"
    )
    assert "vram=" not in status


def test_vram_ballast_writes_every_allocated_byte():
    class Block:
        def __init__(self, size):
            self.size = size
            self.fills = []

        def fill_(self, value):
            self.fills.append(value)

    class FakeCuda:
        def __init__(self):
            self.synchronized = []

        def device(self, _index):
            return nullcontext()

        def mem_get_info(self):
            return 1000, 2000

        def synchronize(self, device):
            self.synchronized.append(device)

    cuda = FakeCuda()
    blocks = []

    def allocate(size, **_kwargs):
        block = Block(size)
        blocks.append(block)
        return block

    fake_torch = SimpleNamespace(
        cuda=cuda,
        uint8=object(),
        empty=allocate,
        OutOfMemoryError=RuntimeError,
    )
    device = object()

    ballast, allocated_bytes = gmark._allocate_vram_ballast(
        fake_torch,
        device,
        0,
        0.5,
    )

    assert ballast == blocks
    assert allocated_bytes == 500
    assert [(block.size, block.fills) for block in blocks] == [(500, [0xA5])]
    assert cuda.synchronized == [device]


def test_integrity_check_rejects_nonfinite_and_changed_results():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    gmark._validate_sample(torch, reference.clone(), reference, 1)

    changed = reference.clone()
    changed[0, 0] = 5.0
    with pytest.raises(gmark.GmarkError, match="sampled output changed"):
        gmark._validate_sample(torch, changed, reference, 1)

    nonfinite = reference.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(gmark.GmarkError, match="non-finite matrix values"):
        gmark._validate_sample(torch, nonfinite, reference, 1)


def test_main_preserves_gmark_exit_status_without_generic_device_resolution(
    monkeypatch,
):
    args = SimpleNamespace(action="gmark")
    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(main_module, "run_gmark", lambda _args: 3)
    monkeypatch.setattr(
        main_module,
        "get_device",
        lambda _device: pytest.fail("generic device resolution must not run"),
    )

    with pytest.raises(SystemExit) as error:
        main_module.main()

    assert error.value.code == 3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device is required for gmark integration",
)
def test_gmark_completes_a_small_real_cuda_workload(capsys):
    args = build_parser().parse_args([
        "gmark",
        "--duration=0.01",
        "--matrix-size=64",
        "--memory-fraction=0",
        "--max-temperature=0",
        "--status-interval=0.005",
        "--warmup-iterations=1",
        "--sync-every=1",
    ])

    result = gmark.run(args)

    assert result == 0
    assert "PASS: requested duration completed" in capsys.readouterr().out
