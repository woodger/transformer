from importlib.metadata import version as package_version

import app.service.adapters.outbound.cuda.inventory as device_inventory_module
from app.service.adapters.outbound.cuda.inventory import (
    CudaDeviceInventory,
    CudaDeviceState,
)


def _probe(*identifiers):
    return {
        "devices": [
            {
                "id": identifier,
                "ordinal": ordinal,
                "name": f"GPU {ordinal}",
            }
            for ordinal, identifier in enumerate(identifiers)
        ],
        "runtimeVersion": "13.3",
        "torchVersion": "2.12.0",
    }


def test_inventory_exposes_one_capacity_unit_per_physical_gpu():
    inventory = CudaDeviceInventory(
        probe=lambda: _probe("GPU-a", "GPU-b"),
    ).initialize()

    snapshot = inventory.snapshot()

    assert snapshot.device_count == 2
    assert snapshot.cuda_capacity == 2
    assert snapshot.quarantined_count == 0
    assert [device.device_id for device in snapshot.devices] == [
        "GPU-a",
        "GPU-b",
    ]
    assert inventory.mark_busy("GPU-a") is True
    assert inventory.snapshot().devices[0].state is CudaDeviceState.BUSY
    # Capacity describes physical scheduler lanes; a busy lease does not
    # make the device disappear from capabilities.
    assert inventory.snapshot().cuda_capacity == 2
    inventory.release("GPU-a")
    assert inventory.snapshot().devices[0].state is CudaDeviceState.AVAILABLE


def test_confirmed_loss_quarantines_device_for_current_inventory():
    probes = iter([
        _probe("GPU-a", "GPU-b"),
        _probe("GPU-b"),
        _probe("GPU-a", "GPU-b"),
    ])
    inventory = CudaDeviceInventory(
        probe=lambda: next(probes),
    ).initialize()

    assert inventory.confirm_loss("GPU-a") is True
    snapshot = inventory.snapshot()
    assert snapshot.cuda_capacity == 1
    assert snapshot.quarantined_count == 1
    assert inventory.is_quarantined("GPU-a") is True

    # Quarantine is monotonic. The third probe is deliberately never
    # consulted.
    assert inventory.confirm_loss("GPU-a") is True
    assert inventory.snapshot().quarantined_count == 1


def test_quarantine_survives_service_restart_in_same_linux_boot(
    tmp_path,
):
    quarantine_path = tmp_path / "cuda-quarantine.json"
    probes = iter([
        _probe("GPU-a", "GPU-b"),
        _probe("GPU-b"),
    ])
    first = CudaDeviceInventory(
        probe=lambda: next(probes),
        quarantine_path=quarantine_path,
    ).initialize()

    assert first.confirm_loss("GPU-a") is True

    restarted = CudaDeviceInventory(
        probe=lambda: _probe("GPU-a", "GPU-b"),
        quarantine_path=quarantine_path,
    ).initialize()
    snapshot = restarted.snapshot()

    assert snapshot.device_count == 2
    assert snapshot.cuda_capacity == 1
    assert snapshot.quarantined_count == 1
    assert restarted.is_quarantined("GPU-a") is True
    assert restarted.mark_busy("GPU-a") is False
    assert restarted.mark_busy("GPU-b") is True


def test_new_linux_boot_rebuilds_inventory_without_old_quarantine(
    tmp_path,
    monkeypatch,
):
    quarantine_path = tmp_path / "cuda-quarantine.json"
    boot_ids = iter([
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ])
    monkeypatch.setattr(
        device_inventory_module,
        "_current_boot_id",
        lambda: next(boot_ids),
    )
    probes = iter([
        _probe("GPU-a"),
        _probe(),
    ])
    first = CudaDeviceInventory(
        probe=lambda: next(probes),
        quarantine_path=quarantine_path,
    ).initialize()
    assert first.confirm_loss("GPU-a") is True

    restarted = CudaDeviceInventory(
        probe=lambda: _probe("GPU-a"),
        quarantine_path=quarantine_path,
    ).initialize()

    assert restarted.snapshot().cuda_capacity == 1
    assert restarted.snapshot().quarantined_count == 0
    assert restarted.mark_busy("GPU-a") is True


def test_failed_device_loss_probe_quarantines_assigned_device():
    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _probe("GPU-a")
        raise RuntimeError("CUDA runtime initialization failed")

    inventory = CudaDeviceInventory(probe=probe).initialize()

    assert inventory.confirm_loss("GPU-a") is True
    assert inventory.snapshot().cuda_capacity == 0


def test_inventory_probe_failure_keeps_cpu_service_startable():
    inventory = CudaDeviceInventory(
        probe=lambda: (_ for _ in ()).throw(
            RuntimeError("driver unavailable")
        ),
    ).initialize()

    assert inventory.snapshot().devices == ()
    assert inventory.snapshot().cuda_capacity == 0


def test_default_probe_is_independent_of_service_working_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    inventory = CudaDeviceInventory().initialize()

    assert inventory.snapshot().torch_version == package_version("torch")
