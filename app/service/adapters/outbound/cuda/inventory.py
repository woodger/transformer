from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v20 import CONTRACT_VERSION, validate_document
from app.project import PROJECT_ROOT
from app.service.application.ports.observability import EventLogger


class CudaDeviceState(StrEnum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class CudaDevice:
    device_id: str
    ordinal: int
    name: str
    state: CudaDeviceState


@dataclass(frozen=True, slots=True)
class CudaInventorySnapshot:
    devices: tuple[CudaDevice, ...]
    runtime_version: str | None
    torch_version: str

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def cuda_capacity(self) -> int:
        return sum(
            device.state != CudaDeviceState.QUARANTINED
            for device in self.devices
        )

    @property
    def quarantined_count(self) -> int:
        return sum(
            device.state == CudaDeviceState.QUARANTINED
            for device in self.devices
        )


class CudaDeviceInventory:
    """Физический inventory CUDA в пределах boot и монотонный карантин."""

    def __init__(
        self,
        *,
        probe: Callable[[], JsonObject] | None = None,
        logger: EventLogger | None = None,
        quarantine_path: str | None = None,
    ) -> None:
        self._probe = probe or _probe_cuda
        self._logger = logger
        self._quarantine_path = (
            None
            if quarantine_path is None
            else os.path.abspath(os.fspath(quarantine_path))
        )
        self._boot_id: str | None = None
        self._lock = threading.Lock()
        self._quarantine_persistence_lock = threading.Lock()
        self._devices: dict[str, CudaDevice] = {}
        self._runtime_version: str | None = None
        self._torch_version = "unknown"

    def initialize(self) -> CudaDeviceInventory:
        try:
            persisted = ()
            if self._quarantine_path is not None:
                self._boot_id = _current_boot_id()
                persisted = _load_quarantine(
                    self._quarantine_path,
                    self._boot_id,
                )
            document = self._probe()
            probe_snapshot = _parse_probe(document)
            devices = _merge_quarantine(probe_snapshot.devices, persisted)
            runtime_version = probe_snapshot.runtime_version
            torch_version = probe_snapshot.torch_version
        except Exception as exc:
            devices = ()
            runtime_version = None
            torch_version = "unknown"
            if self._logger is not None:
                self._logger.event(
                    "flight.cuda.inventory_failed",
                    errorType=type(exc).__name__,
                )
        with self._lock:
            self._devices = {
                device.device_id: device
                for device in devices
            }
            self._runtime_version = runtime_version
            self._torch_version = torch_version
        return self

    def snapshot(self) -> CudaInventorySnapshot:
        with self._lock:
            devices = tuple(sorted(
                self._devices.values(),
                key=lambda item: item.ordinal,
            ))
            return CudaInventorySnapshot(
                devices=devices,
                runtime_version=self._runtime_version,
                torch_version=self._torch_version,
            )

    def schedulable_devices(self) -> tuple[CudaDevice, ...]:
        return tuple(
            device
            for device in self.snapshot().devices
            if device.state != CudaDeviceState.QUARANTINED
        )

    def mark_busy(self, device_id: str) -> bool:
        with self._lock:
            device = self._devices.get(device_id)
            if (
                device is None
                or device.state != CudaDeviceState.AVAILABLE
            ):
                return False
            self._devices[device_id] = CudaDevice(
                device.device_id,
                device.ordinal,
                device.name,
                CudaDeviceState.BUSY,
            )
            return True

    def release(self, device_id: str) -> None:
        with self._lock:
            device = self._devices.get(device_id)
            if (
                device is None
                or device.state == CudaDeviceState.QUARANTINED
            ):
                return
            self._devices[device_id] = CudaDevice(
                device.device_id,
                device.ordinal,
                device.name,
                CudaDeviceState.AVAILABLE,
            )

    def is_quarantined(self, device_id: str) -> bool:
        with self._lock:
            device = self._devices.get(device_id)
            return (
                device is None
                or device.state == CudaDeviceState.QUARANTINED
            )

    def confirm_loss(self, device_id: str) -> bool:
        """Проверить устройство один раз и изолировать пропавшее до следующего boot."""

        with self._lock:
            assigned = self._devices.get(device_id)
            if assigned is None:
                return True
            if assigned.state == CudaDeviceState.QUARANTINED:
                return True
        try:
            document = self._probe()
            probe_snapshot = _parse_probe(document)
            live_ids = {
                device.device_id
                for device in probe_snapshot.devices
            }
            lost = device_id not in live_ids
        except Exception:
            # Проба инвентаря, не способная инициализировать среду CUDA, сама
            # показывает, что процесс должен прекратить назначать выделенное
            # устройство.
            lost = True
        if not lost:
            return False
        with self._lock:
            current = self._devices.get(device_id)
            if current is not None:
                self._devices[device_id] = CudaDevice(
                    current.device_id,
                    current.ordinal,
                    current.name,
                    CudaDeviceState.QUARANTINED,
                )
        if (
            self._quarantine_path is not None
            and self._boot_id is not None
        ):
            try:
                with self._quarantine_persistence_lock:
                    with self._lock:
                        quarantined = tuple(
                            device
                            for device in self._devices.values()
                            if (
                                device.state
                                == CudaDeviceState.QUARANTINED
                            )
                        )
                    _save_quarantine(
                        self._quarantine_path,
                        self._boot_id,
                        quarantined,
                    )
            except OSError as exc:
                if self._logger is not None:
                    self._logger.event(
                        "flight.cuda.quarantine_persistence_failed",
                        errorType=type(exc).__name__,
                    )
        if self._logger is not None:
            self._logger.event(
                "flight.cuda.quarantined",
                deviceId=device_id,
            )
        return True


def static_cuda_inventory(
    available: Callable[[], bool],
) -> CudaDeviceInventory:
    def probe() -> JsonObject:
        if not available():
            return {
                "devices": [],
                "runtimeVersion": None,
                "torchVersion": "unknown",
            }
        return {
            "devices": [{
                "id": "0",
                "ordinal": 0,
                "name": "CUDA",
            }],
            "runtimeVersion": "unknown",
            "torchVersion": "unknown",
        }

    return CudaDeviceInventory(probe=probe).initialize()


def _probe_cuda() -> JsonObject:
    capabilities = _worker_inspect()
    metadata: JsonObject = {
        "runtimeVersion": _optional_string(
            capabilities["cudaRuntimeVersion"],
            "worker CUDA runtime version",
        ),
        "torchVersion": _required_string(
            capabilities["torchVersion"],
            "worker Torch version",
        ),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "devices": [],
            **metadata,
        }
    if result.returncode != 0:
        return {
            "devices": [],
            **metadata,
        }

    physical_devices: list[tuple[int, str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise ValueError("nvidia-smi returned an invalid GPU row")
        index_text, identifier, name = parts
        physical_devices.append(
            (int(index_text), identifier, name)
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selected = {
            value.strip()
            for value in visible.split(",")
            if value.strip()
        }
        physical_devices = [
            device
            for device in physical_devices
            if (
                str(device[0]) in selected
                or device[1] in selected
            )
        ]

    devices: list[JsonValue] = []
    for ordinal, identifier, reported_name in physical_devices:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = identifier
        try:
            document = _worker_inspect(environment)
            visible_devices = document["devices"]
            if not isinstance(visible_devices, list):
                raise ValueError("worker CUDA devices must be an array")
            cuda_devices = [
                device
                for device in visible_devices
                if isinstance(device, dict) and device.get("backend") == "cuda"
            ]
            if len(cuda_devices) != 1:
                continue
            visible_device = cuda_devices[0]
            name = visible_device.get("name")
            if not isinstance(name, str) or not name:
                name = reported_name
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ):
            continue
        devices.append({
            "id": identifier,
            "ordinal": ordinal,
            "name": name,
        })
    return {
        "devices": devices,
        **metadata,
    }


def _worker_inspect(
    environment: dict[str, str] | None = None,
) -> JsonObject:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.worker.bootstrap",
            "inspect",
            f"--contract-version={CONTRACT_VERSION}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=PROJECT_ROOT,
        env=environment,
    )
    try:
        document: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "worker capability response is not valid JSON"
        ) from exc
    return validate_document(document, "capabilities")


def _parse_probe(
    document: JsonObject,
) -> CudaInventorySnapshot:
    raw_devices = document.get("devices")
    runtime_version = document.get("runtimeVersion")
    torch_version = document.get("torchVersion")
    if not isinstance(raw_devices, list):
        raise ValueError("CUDA probe devices must be an array")
    if runtime_version is not None and not isinstance(
        runtime_version,
        str,
    ):
        raise ValueError("CUDA runtime version must be a string or null")
    if not isinstance(torch_version, str) or not torch_version:
        raise ValueError("Torch version must be a non-empty string")
    devices: list[CudaDevice] = []
    identifiers: set[str] = set()
    ordinals: set[int] = set()
    for value in raw_devices:
        if not isinstance(value, dict) or set(value) != {
            "id",
            "ordinal",
            "name",
        }:
            raise ValueError("CUDA probe device is invalid")
        typed_value = cast(JsonObject, value)
        identifier = typed_value["id"]
        ordinal = typed_value["ordinal"]
        name = typed_value["name"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("CUDA device ID is invalid")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or ordinal in ordinals
        ):
            raise ValueError("CUDA device ordinal is invalid")
        if not isinstance(name, str) or not name:
            raise ValueError("CUDA device name is invalid")
        identifiers.add(identifier)
        ordinals.add(ordinal)
        devices.append(CudaDevice(
            identifier,
            ordinal,
            name,
            CudaDeviceState.AVAILABLE,
        ))
    return CudaInventorySnapshot(
        devices=tuple(devices),
        runtime_version=runtime_version,
        torch_version=torch_version,
    )


def _current_boot_id() -> str:
    with open(
        "/proc/sys/kernel/random/boot_id",
        encoding="ascii",
    ) as source:
        value = source.read().strip()
    parsed = uuid.UUID(value)
    if str(parsed) != value.lower():
        raise ValueError("Linux boot ID is not canonical")
    return str(parsed)


def _load_quarantine(
    path: str,
    boot_id: str,
) -> tuple[CudaDevice, ...]:
    try:
        with open(path, encoding="utf-8") as source:
            document: object = json.load(source)
    except FileNotFoundError:
        return ()
    if not isinstance(document, dict):
        raise ValueError("CUDA quarantine document is invalid")
    typed_document = cast(JsonObject, document)
    if set(typed_document) != {"bootId", "devices"}:
        raise ValueError("CUDA quarantine document is invalid")
    raw_boot_id = typed_document["bootId"]
    if not isinstance(raw_boot_id, str):
        raise ValueError("CUDA quarantine boot ID is invalid")
    try:
        document_boot_id = str(uuid.UUID(raw_boot_id))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "CUDA quarantine boot ID is invalid"
        ) from exc
    if document_boot_id != raw_boot_id.lower():
        raise ValueError("CUDA quarantine boot ID is not canonical")
    if document_boot_id != boot_id:
        return ()
    devices = typed_document["devices"]
    if not isinstance(devices, list):
        raise ValueError("CUDA quarantine devices must be an array")
    parsed: list[CudaDevice] = []
    for value in devices:
        if not isinstance(value, dict) or set(value) != {
            "id",
            "ordinal",
            "name",
        }:
            raise ValueError("CUDA quarantine device is invalid")
        typed_value = cast(JsonObject, value)
        identifier = typed_value["id"]
        ordinal = typed_value["ordinal"]
        name = typed_value["name"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError("CUDA quarantine device fields are invalid")
        parsed.append(CudaDevice(
            identifier,
            ordinal,
            name,
            CudaDeviceState.QUARANTINED,
        ))
    if (
        len({device.device_id for device in parsed}) != len(parsed)
        or len({device.ordinal for device in parsed}) != len(parsed)
    ):
        raise ValueError("CUDA quarantine devices are not unique")
    return tuple(parsed)


def _required_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: JsonValue, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")
    return value


def _merge_quarantine(
    discovered: tuple[CudaDevice, ...],
    persisted: tuple[CudaDevice, ...],
) -> tuple[CudaDevice, ...]:
    devices = {
        device.device_id: device
        for device in discovered
    }
    used_ordinals = {
        device.ordinal: device.device_id
        for device in discovered
    }
    for saved in persisted:
        current = devices.get(saved.device_id)
        if current is not None:
            devices[saved.device_id] = CudaDevice(
                current.device_id,
                current.ordinal,
                current.name,
                CudaDeviceState.QUARANTINED,
            )
            continue
        if (
            saved.ordinal in used_ordinals
            and used_ordinals[saved.ordinal] != saved.device_id
        ):
            saved = CudaDevice(
                saved.device_id,
                max(used_ordinals, default=-1) + 1,
                saved.name,
                saved.state,
            )
        devices[saved.device_id] = saved
        used_ordinals[saved.ordinal] = saved.device_id
    return tuple(devices.values())


def _save_quarantine(
    path: str,
    boot_id: str,
    devices: tuple[CudaDevice, ...],
) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    document = {
        "bootId": boot_id,
        "devices": [
            {
                "id": device.device_id,
                "ordinal": device.ordinal,
                "name": device.name,
            }
            for device in sorted(
                devices,
                key=lambda item: item.ordinal,
            )
        ],
    }
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "CudaDevice",
    "CudaDeviceInventory",
    "CudaDeviceState",
    "CudaInventorySnapshot",
    "static_cuda_inventory",
]
