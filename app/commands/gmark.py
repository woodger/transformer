from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from typing import Any

MIB = 1024**2
GIB = 1024**3
BALLAST_CHUNK_BYTES = 256 * MIB


class GmarkError(RuntimeError):
    """Expected CUDA, monitoring, configuration or integrity failure."""


@dataclasses.dataclass(frozen=True)
class GpuMetrics:
    temperature_c: float | None
    memory_temperature_c: float | None
    utilization_percent: float | None
    power_w: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None


class NvidiaSmiMonitor:
    """Best-effort metrics reader without a Python NVML dependency."""

    def __init__(self, device_identifiers: Sequence[str]) -> None:
        self._executable = shutil.which("nvidia-smi")
        self._device_identifiers = tuple(dict.fromkeys(device_identifiers))

    def read(self) -> GpuMetrics | None:
        if self._executable is None:
            return None

        for device_identifier in self._device_identifiers:
            values = self._query(
                device_identifier,
                "temperature.gpu,temperature.memory,utilization.gpu,"
                "power.draw,memory.used,memory.total",
                expected_fields=6,
            )
            if values is not None:
                return GpuMetrics(*values)

            # Consumer cards and some drivers omit the optional VRAM sensor.
            values = self._query(
                device_identifier,
                "temperature.gpu,utilization.gpu,power.draw,"
                "memory.used,memory.total",
                expected_fields=5,
            )
            if values is not None:
                return GpuMetrics(values[0], None, *values[1:])
        return None

    def _query(
        self,
        device_identifier: str,
        fields: str,
        expected_fields: int,
    ) -> list[float | None] | None:
        assert self._executable is not None
        try:
            result = subprocess.run(
                [
                    self._executable,
                    f"--id={device_identifier}",
                    f"--query-gpu={fields}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        output_fields = [
            field.strip() for field in result.stdout.strip().split(",")
        ]
        if len(output_fields) != expected_fields:
            return None
        return [_optional_float(field) for field in output_fields]


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _cuda_memory_info(torch: Any, device_index: int) -> tuple[int, int]:
    with torch.cuda.device(device_index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def _allocate_vram_ballast(
    torch: Any,
    device: Any,
    device_index: int,
    fraction: float,
) -> tuple[list[Any], int]:
    if fraction == 0:
        return [], 0

    free_bytes, _ = _cuda_memory_info(torch, device_index)
    target_bytes = int(free_bytes * fraction)
    ballast: list[Any] = []
    allocated_bytes = 0

    while allocated_bytes < target_bytes:
        allocation_bytes = min(
            BALLAST_CHUNK_BYTES,
            target_bytes - allocated_bytes,
        )
        try:
            block = torch.empty(
                allocation_bytes,
                dtype=torch.uint8,
                device=device,
            )
            # Touch every byte so the workload reaches physical VRAM.
            block.fill_(0xA5)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                "Warning: VRAM allocation stopped before its target due to OOM.",
                file=sys.stderr,
                flush=True,
            )
            break
        ballast.append(block)
        allocated_bytes += allocation_bytes

    torch.cuda.synchronize(device)
    return ballast, allocated_bytes


def _monitor_identifiers(
    properties: Any,
    fallback_index: int,
) -> tuple[str, ...]:
    identifier = getattr(properties, "uuid", None)
    if isinstance(identifier, bytes):
        if len(identifier) == 16:
            identifier = str(uuid.UUID(bytes=identifier))
        else:
            identifier = identifier.decode("ascii", errors="replace")
    if identifier:
        identifier_text = str(identifier)
        if not identifier_text.startswith(("GPU-", "MIG-")):
            identifier_text = f"GPU-{identifier_text}"
        return identifier_text, str(fallback_index)
    return (str(fallback_index),)


def _format_gib(byte_count: int) -> str:
    return f"{byte_count / GIB:.2f} GiB"


def _format_status(
    elapsed: float,
    iterations: int,
    tflops: float,
    metrics: GpuMetrics | None,
) -> str:
    parts = [
        f"[{elapsed:7.1f}s]",
        f"iterations={iterations}",
        f"average={tflops:.2f} TFLOP/s",
    ]
    if metrics is not None:
        if metrics.temperature_c is not None:
            parts.append(f"gpu-temp={metrics.temperature_c:.0f} C")
        if metrics.memory_temperature_c is not None:
            parts.append(f"vram-temp={metrics.memory_temperature_c:.0f} C")
        if metrics.utilization_percent is not None:
            parts.append(f"util={metrics.utilization_percent:.0f}%")
        if metrics.power_w is not None:
            parts.append(f"power={metrics.power_w:.1f} W")
        if (
            metrics.memory_used_mib is not None
            and metrics.memory_total_mib is not None
        ):
            parts.append(
                "vram="
                f"{metrics.memory_used_mib / 1024:.2f}/"
                f"{metrics.memory_total_mib / 1024:.2f} GiB"
            )
    return " | ".join(parts)


def _validate_sample(
    torch: Any,
    output: Any,
    reference: Any,
    step: int,
) -> None:
    current = output[::step, ::step]
    if not bool(torch.isfinite(current).all().item()):
        raise GmarkError("integrity check failed: non-finite matrix values")
    if not bool(torch.allclose(current, reference, rtol=1e-3, atol=1e-3)):
        max_error = float(
            (current.float() - reference.float()).abs().max().item()
        )
        raise GmarkError(
            "integrity check failed: sampled output changed "
            f"(max error {max_error:g})"
        )


def _require_temperature(
    metrics: GpuMetrics | None,
    max_temperature: float,
    *,
    during_workload: bool,
) -> float | None:
    if max_temperature == 0:
        return metrics.temperature_c if metrics is not None else None
    if metrics is None or metrics.temperature_c is None:
        suffix = "; workload stopped" if during_workload else ""
        raise GmarkError(
            "GPU temperature is unavailable from nvidia-smi, so the thermal "
            f"cutoff cannot be enforced{suffix}"
        )
    return metrics.temperature_c


def _update_peak(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None else max(current, value)


def run_stress_test(torch: Any, args: Any) -> int:
    if not torch.cuda.is_available():
        cuda_build = getattr(torch.version, "cuda", None) or "none"
        raise GmarkError(
            "CUDA is not available to PyTorch "
            f"(PyTorch CUDA build: {cuda_build})"
        )
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise GmarkError(
            f"CUDA device {args.device} does not exist; visible device count is "
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise GmarkError(
            f"{properties.name} does not support CUDA bfloat16 operations"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    free_before, total_bytes = _cuda_memory_info(torch, args.device)
    print(f"GPU: {properties.name} (cuda:{args.device})")
    print(
        f"Compute capability: {properties.major}.{properties.minor} | "
        f"VRAM: {_format_gib(total_bytes)} | free: {_format_gib(free_before)}"
    )
    print(
        f"Workload: {args.matrix_size} x {args.matrix_size} {args.dtype} "
        f"matrix multiplication | duration: {args.duration:g}s"
    )
    if args.max_temperature > 0:
        print(
            f"Thermal cutoff: {args.max_temperature:g} C "
            "(nvidia-smi GPU sensor; VRAM/VRM sensors may not be exposed)"
        )
    else:
        print("Thermal cutoff: disabled")

    monitor = NvidiaSmiMonitor(_monitor_identifiers(properties, args.device))
    metrics = monitor.read()
    initial_temperature = _require_temperature(
        metrics,
        args.max_temperature,
        during_workload=False,
    )
    if metrics is None:
        print("Warning: nvidia-smi metrics are unavailable.", file=sys.stderr)
    if (
        args.max_temperature > 0
        and initial_temperature is not None
        and initial_temperature >= args.max_temperature
    ):
        raise GmarkError(
            f"GPU is already at {initial_temperature:.0f} C, at or above the "
            f"{args.max_temperature:g} C cutoff"
        )

    try:
        matrix_a = torch.randn(
            (args.matrix_size, args.matrix_size),
            dtype=dtype,
            device=device,
        )
        matrix_b = torch.randn(
            (args.matrix_size, args.matrix_size),
            dtype=dtype,
            device=device,
        )
        output = torch.empty_like(matrix_a)
        for _ in range(args.warmup_iterations):
            torch.mm(matrix_a, matrix_b, out=output)
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        raise GmarkError(
            "not enough VRAM for the matrices; reduce --matrix-size"
        ) from error

    sample_step = max(1, args.matrix_size // 64)
    reference = output[::sample_step, ::sample_step].clone()
    _validate_sample(torch, output, reference, sample_step)

    ballast, ballast_bytes = _allocate_vram_ballast(
        torch,
        device,
        args.device,
        args.memory_fraction,
    )
    free_after, _ = _cuda_memory_info(torch, args.device)
    print(
        f"VRAM ballast: {_format_gib(ballast_bytes)} written | "
        f"free after setup: {_format_gib(free_after)}"
    )

    metrics = monitor.read()
    setup_temperature = _require_temperature(
        metrics,
        args.max_temperature,
        during_workload=False,
    )
    if (
        args.max_temperature > 0
        and setup_temperature is not None
        and setup_temperature >= args.max_temperature
    ):
        raise GmarkError(
            f"GPU is already at {setup_temperature:.0f} C, at or above the "
            f"{args.max_temperature:g} C cutoff"
        )

    print("Stress test started. Press Ctrl+C to stop.", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.monotonic()
    deadline = started_at + args.duration
    next_status_at = started_at + args.status_interval
    iterations = 0
    stopped_for_temperature = False
    interrupted = False
    ballast_index = 0
    peak_temperature = setup_temperature
    peak_memory_temperature = (
        metrics.memory_temperature_c if metrics is not None else None
    )
    peak_power = metrics.power_w if metrics is not None else None
    operations_per_iteration = 2 * args.matrix_size**3

    try:
        while time.monotonic() < deadline:
            for _ in range(args.sync_every):
                torch.mm(matrix_a, matrix_b, out=output)
                iterations += 1
            if ballast:
                # Cycling one chunk per sync keeps reserved VRAM under load.
                ballast[ballast_index].bitwise_xor_(0xFF)
                ballast_index = (ballast_index + 1) % len(ballast)
            torch.cuda.synchronize(device)

            now = time.monotonic()
            if now >= next_status_at or now >= deadline:
                elapsed = now - started_at
                _validate_sample(torch, output, reference, sample_step)
                metrics = monitor.read()
                temperature = _require_temperature(
                    metrics,
                    args.max_temperature,
                    during_workload=True,
                )
                if metrics is not None:
                    peak_temperature = _update_peak(
                        peak_temperature,
                        metrics.temperature_c,
                    )
                    peak_memory_temperature = _update_peak(
                        peak_memory_temperature,
                        metrics.memory_temperature_c,
                    )
                    peak_power = _update_peak(peak_power, metrics.power_w)
                tflops = operations_per_iteration * iterations / elapsed / 1e12
                print(
                    _format_status(elapsed, iterations, tflops, metrics),
                    flush=True,
                )
                next_status_at = now + args.status_interval

                if (
                    args.max_temperature > 0
                    and temperature is not None
                    and temperature >= args.max_temperature
                ):
                    stopped_for_temperature = True
                    print(
                        "Temperature cutoff reached: "
                        f"{temperature:.0f} C >= "
                        f"{args.max_temperature:g} C.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user.", file=sys.stderr, flush=True)

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started_at
    _validate_sample(torch, output, reference, sample_step)
    average_tflops = (
        operations_per_iteration * iterations / elapsed / 1e12
        if elapsed > 0
        else 0.0
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    print(
        f"Result: {iterations} iterations in {elapsed:.1f}s | "
        f"average {average_tflops:.2f} TFLOP/s | "
        f"PyTorch peak allocation {_format_gib(peak_allocated)}"
    )
    thermal_summary: list[str] = []
    if peak_temperature is not None:
        thermal_summary.append(f"peak GPU temperature {peak_temperature:.0f} C")
    if peak_memory_temperature is not None:
        thermal_summary.append(
            f"peak VRAM temperature {peak_memory_temperature:.0f} C"
        )
    if peak_power is not None:
        thermal_summary.append(f"peak power {peak_power:.1f} W")
    if thermal_summary:
        print("Thermals: " + " | ".join(thermal_summary))

    if stopped_for_temperature:
        print("FAIL: stopped by the temperature safety cutoff.")
        return 3
    if interrupted:
        print("STOPPED: workload and sampled-output checks were clean.")
        return 130
    print("PASS: requested duration completed and sampled-output checks were clean.")
    return 0


def run(args: Any, *, torch_module: Any | None = None) -> int:
    if torch_module is None:
        import torch

        torch_module = torch
    try:
        return run_stress_test(torch_module, args)
    except GmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"ERROR: CUDA workload failed: {error}", file=sys.stderr)
        return 1


__all__ = [
    "GmarkError",
    "GpuMetrics",
    "NvidiaSmiMonitor",
    "run",
    "run_stress_test",
]
