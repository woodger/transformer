from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from app.config import (
    CONTEXT_MODE,
    DROPOUT,
    GRAD_CLIP_NORM,
    LOSS_STAGE,
    LR,
    WEIGHT_DECAY,
)

MIB = 1024**2
GIB = 1024**3
BALLAST_CHUNK_BYTES = 256 * MIB
MAX_AMP_SCALE_BACKOFFS = 32


class GmarkError(RuntimeError):
    """Expected CUDA, monitoring, configuration or integrity failure."""


@dataclasses.dataclass(frozen=True)
class GpuMetrics:
    temperature_c: float | None
    memory_temperature_c: float | None
    utilization_percent: float | None
    power_w: float | None


@dataclasses.dataclass(frozen=True)
class TrainingStep:
    loss: float
    grad_norm: float


class TrainingWorkload:
    """Synthetic batch driven through the production training primitives."""

    def __init__(self, torch: Any, args: Any, device: Any) -> None:
        from app.worker.model.transformer import TransformerModel
        from app.worker.training.losses import combined_loss

        self._torch = torch
        self._device = device
        self._combined_loss = combined_loss
        self._use_amp = bool(args.use_amp)
        self._batch_size = args.batch_size
        self._amp_backoffs = 0

        self.model = TransformerModel(
            input_dim=args.feature_dim,
            seq_len=args.seq_len,
            hidden_dim=args.hidden,
            layers=args.layers,
            dropout=DROPOUT,
            out_dim=6,
            nhead=args.nhead,
            context_mode=CONTEXT_MODE,
        ).to(device)
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.scaler = torch.amp.GradScaler(enabled=self._use_amp)

        self._source = torch.randn(
            (args.batch_size, args.seq_len, args.feature_dim),
            dtype=torch.float32,
        )
        self._targets = torch.zeros(
            (args.batch_size, 6),
            dtype=torch.float32,
        )
        self._targets[:, 0] = (
            torch.randn(args.batch_size, dtype=torch.float32) * 0.05
        )
        self._targets[:, 4] = (
            torch.rand(args.batch_size, dtype=torch.float32) * 0.2 + 1e-3
        )
        self._targets[:, 5] = torch.randint(
            0,
            2,
            (args.batch_size,),
            dtype=torch.int64,
        ).to(dtype=torch.float32)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def amp_scale(self) -> float | None:
        if not self._use_amp:
            return None
        return float(self.scaler.get_scale())

    @property
    def amp_backoffs(self) -> int:
        return self._amp_backoffs

    def _autocast(self) -> AbstractContextManager[object]:
        if self._use_amp:
            return self._torch.amp.autocast(
                device_type="cuda",
                enabled=True,
            )
        return nullcontext()

    def step(self) -> TrainingStep:
        for _ in range(MAX_AMP_SCALE_BACKOFFS):
            result = self._attempt_step()
            if result is not None:
                return result
        raise GmarkError(
            "AMP gradient overflow did not recover after "
            f"{MAX_AMP_SCALE_BACKOFFS} scale backoffs"
        )

    def _attempt_step(self) -> TrainingStep | None:
        torch = self._torch
        source = self._source.to(self._device)
        targets = self._targets.to(self._device)

        self.optimizer.zero_grad()
        with self._autocast():
            predictions = self.model(source)
            loss = self._combined_loss(
                predictions,
                targets,
                LOSS_STAGE,
            )

        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise GmarkError("integrity check failed: non-finite training loss")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            GRAD_CLIP_NORM,
        )
        grad_norm_value = float(grad_norm.detach().cpu())
        grad_norm_is_finite = math.isfinite(grad_norm_value)
        if not grad_norm_is_finite and not self._use_amp:
            raise GmarkError(
                "integrity check failed: non-finite gradient norm"
            )

        scale_before = self.amp_scale
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if not grad_norm_is_finite:
            scale_after = self.amp_scale
            if (
                scale_before is not None
                and scale_after is not None
                and scale_after < scale_before
            ):
                self._amp_backoffs += 1
                return None
            raise GmarkError(
                "integrity check failed: non-finite gradient norm"
            )
        return TrainingStep(loss=loss_value, grad_norm=grad_norm_value)

    def validate(self) -> None:
        torch = self._torch
        for name, parameter in self.model.named_parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise GmarkError(
                    "integrity check failed: non-finite model parameter "
                    f"{name}"
                )
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise GmarkError(
                        "integrity check failed: non-finite optimizer state"
                    )


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
                "power.draw",
                expected_fields=4,
            )
            if values is not None:
                return GpuMetrics(*values)

            # Consumer cards and some drivers omit the optional VRAM sensor.
            values = self._query(
                device_identifier,
                "temperature.gpu,utilization.gpu,power.draw",
                expected_fields=3,
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
    steps: int,
    rows: int,
    step: TrainingStep,
    metrics: GpuMetrics | None,
) -> str:
    step_rate = steps / elapsed if elapsed > 0 else 0.0
    row_rate = rows / elapsed if elapsed > 0 else 0.0
    parts = [
        f"[{elapsed:7.1f}s]",
        f"steps={steps}",
        f"average={step_rate:.2f} steps/s",
        f"rows={row_rate:.0f} rows/s",
        f"loss={step.loss:.6g}",
        f"grad-norm={step.grad_norm:.6g}",
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
    return " | ".join(parts)


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


def run_stress_test(
    torch: Any,
    args: Any,
    *,
    workload_factory: Callable[[Any, Any, Any], Any] | None = None,
) -> int:
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    free_before, total_bytes = _cuda_memory_info(torch, args.device)
    print(f"GPU: {properties.name} (cuda:{args.device})")
    print(
        f"Compute capability: {properties.major}.{properties.minor} | "
        f"VRAM: {_format_gib(total_bytes)} | free: {_format_gib(free_before)}"
    )
    print(
        "Training: "
        f"batch={args.batch_size} | seq-len={args.seq_len} | "
        f"feature-dim={args.feature_dim} | hidden={args.hidden} | "
        f"layers={args.layers} | nhead={args.nhead} | dtype=float32 | "
        f"amp={'on' if args.use_amp else 'off'} | duration={args.duration:g}s"
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

    if workload_factory is None:
        workload_factory = TrainingWorkload
    try:
        workload = workload_factory(torch, args, device)
        for _ in range(args.warmup_steps):
            workload.step()
        workload.validate()
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        raise GmarkError(
            "not enough VRAM for the training workload; reduce --batch-size, "
            "--seq-len, --feature-dim or model dimensions"
        ) from error

    amp_scale = getattr(workload, "amp_scale", None)
    if amp_scale is not None:
        print(
            f"AMP warm-up: scale={amp_scale:g} | "
            f"backoffs={getattr(workload, 'amp_backoffs', 0)}"
        )

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

    print("Training stress test started. Press Ctrl+C to stop.", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.monotonic()
    deadline = started_at + args.duration
    next_status_at = started_at + args.status_interval
    steps = 0
    stopped_for_temperature = False
    interrupted = False
    ballast_index = 0
    peak_temperature = setup_temperature
    peak_memory_temperature = (
        metrics.memory_temperature_c if metrics is not None else None
    )
    peak_power = metrics.power_w if metrics is not None else None
    latest_step: TrainingStep | None = None

    try:
        while time.monotonic() < deadline:
            latest_step = workload.step()
            steps += 1
            if ballast:
                # Cycling one chunk per step keeps reserved VRAM under load.
                ballast[ballast_index].bitwise_xor_(0xFF)
                ballast_index = (ballast_index + 1) % len(ballast)
            torch.cuda.synchronize(device)

            now = time.monotonic()
            if now >= next_status_at or now >= deadline:
                elapsed = now - started_at
                workload.validate()
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
                assert latest_step is not None
                print(
                    _format_status(
                        elapsed,
                        steps,
                        steps * workload.batch_size,
                        latest_step,
                        metrics,
                    ),
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
    workload.validate()
    average_steps = steps / elapsed if elapsed > 0 else 0.0
    average_rows = steps * workload.batch_size / elapsed if elapsed > 0 else 0.0
    peak_allocated = torch.cuda.max_memory_allocated(device)
    print(
        f"Result: {steps} optimizer steps in {elapsed:.1f}s | "
        f"average {average_steps:.2f} steps/s | "
        f"{average_rows:.0f} rows/s | "
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
        print("STOPPED: workload and integrity checks were clean.")
        return 130
    print("PASS: requested duration completed and integrity checks were clean.")
    return 0


def run(
    args: Any,
    *,
    torch_module: Any | None = None,
    workload_factory: Callable[[Any, Any, Any], Any] | None = None,
) -> int:
    if torch_module is None:
        import torch

        torch_module = torch
    try:
        return run_stress_test(
            torch_module,
            args,
            workload_factory=workload_factory,
        )
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
    "TrainingStep",
    "TrainingWorkload",
    "run",
    "run_stress_test",
]
