from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.ml import target_identity
from app.worker.model.transformer import public_predictions
from app.worker.training.epoch import DIRECT_LOSS_FIELDS, TrainingEpochResult

_SEMANTICS = (
    "mean_return",
    "sigma_return",
    "prob_tp",
    "prob_sl",
    "volatility_next",
    "hitting_prob_tp",
)


def _empty_float_list() -> list[float]:
    return []


@dataclass(frozen=True, slots=True)
class TargetErrorObservation:
    """Device-resident per-target errors awaiting the batch scalar transfer."""

    values: tuple[torch.Tensor, ...]

    @classmethod
    def evaluate(
        cls,
        model_output: torch.Tensor,
        targets: torch.Tensor,
    ) -> TargetErrorObservation:
        predictions = public_predictions(model_output).detach().float()
        errors = predictions - targets.detach().float()
        absolute_errors = errors.abs()
        squared_errors = errors.square()
        return cls(values=(
            *(absolute_errors[:, index].mean() for index in range(6)),
            *(squared_errors[:, index].mean() for index in range(6)),
        ))

    def decode(self, values: tuple[float, ...]) -> dict[str, float]:
        if len(values) != 12:
            raise ValueError("target error observation must contain 12 values")
        return {
            **{
                f"{semantic}_mae": values[index]
                for index, semantic in enumerate(_SEMANTICS)
            },
            **{
                f"{semantic}_mse": values[index + len(_SEMANTICS)]
                for index, semantic in enumerate(_SEMANTICS)
            },
        }


@dataclass
class EpochTelemetry:
    """Optional runtime observations for one completed training epoch."""

    mean_return_mae: float = 0.0
    sigma_return_mae: float = 0.0
    prob_tp_mae: float = 0.0
    prob_sl_mae: float = 0.0
    volatility_next_mae: float = 0.0
    hitting_prob_tp_mae: float = 0.0
    mean_return_rmse: float = 0.0
    sigma_return_rmse: float = 0.0
    prob_tp_rmse: float = 0.0
    prob_sl_rmse: float = 0.0
    volatility_next_rmse: float = 0.0
    hitting_prob_tp_rmse: float = 0.0
    training_batches_completed: int = 0
    optimizer_updates_applied: int = 0
    optimizer_updates_skipped: int = 0
    amp_overflow_batches: int = 0
    finite_gradient_batches: int = 0
    non_finite_gradient_batches: int = 0
    pre_clip_gradient_norm_mean: float | None = None
    pre_clip_gradient_norm_max: float | None = None
    pre_clip_gradient_norm_p95: float | None = None
    nan_ratio: float = 0.0
    masked_token_ratio: float = 0.0
    complete_token_ratio: float = 0.0
    partial_token_ratio: float = 0.0
    empty_token_ratio: float = 0.0
    input_pipeline_ms: float = 0.0
    missing_stats_ms: float = 0.0
    host_to_device_ms: float = 0.0
    train_step_ms: float = 0.0
    elapsed_ms: float = 0.0
    _rows: int = field(default=0, repr=False)
    _finite_gradient_norms: list[float] = field(
        default_factory=_empty_float_list,
        repr=False,
    )

    def observe_batch(
        self,
        *,
        rows: int,
        target_errors: dict[str, float],
        grad_norm: float | None,
        optimizer_update_applied: bool,
        amp_overflow: bool,
        nan_ratio: float,
        masked_token_ratio: float = 0.0,
        complete_token_ratio: float = 0.0,
        partial_token_ratio: float = 0.0,
        empty_token_ratio: float = 0.0,
    ) -> None:
        total_rows = self._rows + rows
        if total_rows <= 0:
            return

        def average(current: float, value: float) -> float:
            return math.fsum((current * self._rows, value * rows)) / total_rows

        for semantic in _SEMANTICS:
            mae_name = f"{semantic}_mae"
            setattr(
                self,
                mae_name,
                average(getattr(self, mae_name), target_errors[mae_name]),
            )
            rmse_name = f"{semantic}_rmse"
            mse_name = f"{semantic}_mse"
            mse = math.fsum((
                getattr(self, rmse_name) ** 2 * self._rows,
                target_errors[mse_name] * rows,
            )) / total_rows
            setattr(self, rmse_name, math.sqrt(max(0.0, mse)))

        finite_gradient_norm = (
            grad_norm
            if (
                grad_norm is not None
                and math.isfinite(grad_norm)
                and grad_norm >= 0
            )
            else None
        )
        self.training_batches_completed += 1
        if optimizer_update_applied:
            self.optimizer_updates_applied += 1
        else:
            self.optimizer_updates_skipped += 1
        if amp_overflow:
            self.amp_overflow_batches += 1
        if finite_gradient_norm is not None:
            self.finite_gradient_batches += 1
            self._finite_gradient_norms.append(finite_gradient_norm)
        else:
            self.non_finite_gradient_batches += 1
        self.nan_ratio = average(self.nan_ratio, nan_ratio)
        self.masked_token_ratio = average(
            self.masked_token_ratio,
            masked_token_ratio,
        )
        self.complete_token_ratio = average(
            self.complete_token_ratio,
            complete_token_ratio,
        )
        self.partial_token_ratio = average(
            self.partial_token_ratio,
            partial_token_ratio,
        )
        self.empty_token_ratio = average(
            self.empty_token_ratio,
            empty_token_ratio,
        )
        self._rows = total_rows

    def finalize_gradient_statistics(self) -> None:
        if self._finite_gradient_norms:
            ordered = sorted(self._finite_gradient_norms)
            self.pre_clip_gradient_norm_mean = math.fsum(ordered) / len(ordered)
            self.pre_clip_gradient_norm_max = ordered[-1]
            nearest_rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
            self.pre_clip_gradient_norm_p95 = ordered[nearest_rank]
        elif self.finite_gradient_batches == 0:
            self.pre_clip_gradient_norm_mean = None
            self.pre_clip_gradient_norm_max = None
            self.pre_clip_gradient_norm_p95 = None


@dataclass
class ObservedTrainingEpoch(TrainingEpochResult):
    """Core epoch result accompanied by optional best-effort observations."""

    telemetry: EpochTelemetry | None = None


def epoch_telemetry_document(
    result: ObservedTrainingEpoch,
    **extra: JsonValue,
) -> JsonObject | None:
    telemetry = result.telemetry
    if telemetry is None:
        return None
    telemetry.finalize_gradient_statistics()
    return {
        **extra,
        "rows": result.rows,
        "batches": result.batches,
        "loss": result.loss,
        "directLosses": [
            {
                "target": target_identity(index),
                "value": getattr(result, field_name),
            }
            for index, field_name in enumerate(DIRECT_LOSS_FIELDS)
        ],
        "loss_nll": result.loss_nll,
        "loss_ev": result.loss_ev,
        "targetMetrics": [
            {
                "target": target_identity(index),
                "mae": getattr(telemetry, f"{semantic}_mae"),
                "rmse": getattr(telemetry, f"{semantic}_rmse"),
            }
            for index, semantic in enumerate(_SEMANTICS)
        ],
        "selection_score": result.selection_score,
        "trainingBatchesCompleted": telemetry.training_batches_completed,
        "optimizerUpdatesApplied": telemetry.optimizer_updates_applied,
        "optimizerUpdatesSkipped": telemetry.optimizer_updates_skipped,
        "ampOverflowBatches": telemetry.amp_overflow_batches,
        "finiteGradientBatches": telemetry.finite_gradient_batches,
        "nonFiniteGradientBatches": telemetry.non_finite_gradient_batches,
        "preClipGradientNormMean": telemetry.pre_clip_gradient_norm_mean,
        "preClipGradientNormMax": telemetry.pre_clip_gradient_norm_max,
        "preClipGradientNormP95": telemetry.pre_clip_gradient_norm_p95,
        "nan_ratio": telemetry.nan_ratio,
        "masked_token_ratio": telemetry.masked_token_ratio,
        "complete_token_ratio": telemetry.complete_token_ratio,
        "partial_token_ratio": telemetry.partial_token_ratio,
        "empty_token_ratio": telemetry.empty_token_ratio,
        "input_pipeline_ms": telemetry.input_pipeline_ms,
        "missing_stats_ms": telemetry.missing_stats_ms,
        "host_to_device_ms": telemetry.host_to_device_ms,
        "train_step_ms": telemetry.train_step_ms,
        "elapsed_ms": telemetry.elapsed_ms,
        "step": result.step,
        "lr": result.lr,
        "loss_stage": result.loss_stage,
        "minimum_loss_stage": result.minimum_loss_stage,
        "maximum_loss_stage": result.maximum_loss_stage,
    }


def format_epoch_log_line(
    result: ObservedTrainingEpoch,
    **extra: object,
) -> str:
    telemetry = result.telemetry
    gradient_mean = None
    if telemetry is not None:
        telemetry.finalize_gradient_statistics()
        gradient_mean = telemetry.pre_clip_gradient_norm_mean
    fields: dict[str, object] = {
        **extra,
        "loss": f"{result.loss:.6f}",
        **{
            name: f"{getattr(result, name):.6f}"
            for name in DIRECT_LOSS_FIELDS
        },
        "nll": f"{result.loss_nll:.6f}",
        "ev": f"{result.loss_ev:.6f}",
        "grad_mean": "n/a" if gradient_mean is None else f"{gradient_mean:.3f}",
        "rows": result.rows,
        "batches": result.batches,
        "nan": "n/a" if telemetry is None else f"{telemetry.nan_ratio:.4f}",
        "masked_tokens": (
            "n/a" if telemetry is None else f"{telemetry.masked_token_ratio:.4f}"
        ),
        "complete_tokens": (
            "n/a" if telemetry is None else f"{telemetry.complete_token_ratio:.4f}"
        ),
        "partial_tokens": (
            "n/a" if telemetry is None else f"{telemetry.partial_token_ratio:.4f}"
        ),
        "empty_tokens": (
            "n/a" if telemetry is None else f"{telemetry.empty_token_ratio:.4f}"
        ),
        "step": result.step,
        "lr": f"{result.lr:.6g}",
        "loss_stage": result.loss_stage,
        "minimum_loss_stage": result.minimum_loss_stage,
        "maximum_loss_stage": result.maximum_loss_stage,
        "ms": "n/a" if telemetry is None else f"{telemetry.elapsed_ms:.0f}",
    }
    return " ".join(f"{key}={value}" for key, value in fields.items())


def format_epoch_console_line(
    result: ObservedTrainingEpoch,
    **extra: object,
) -> str:
    telemetry = result.telemetry
    if telemetry is not None:
        telemetry.finalize_gradient_statistics()
    fields: list[str] = []
    for key in ("frame", "epoch"):
        if key in extra and extra[key] is not None:
            fields.append(f"{key}={extra[key]}")

    selection_score = extra.get("selection_score")
    selection_text = (
        f"{selection_score:.6g}"
        if isinstance(selection_score, (int, float))
        and math.isfinite(selection_score)
        else "n/a"
    )
    max_loss_stage = extra.get("max_loss_stage", result.loss_stage)

    def observed(name: str, precision: str = ".6g") -> str:
        if telemetry is None:
            return "n/a"
        return format(getattr(telemetry, name), precision)

    gradient_mean = (
        None if telemetry is None else telemetry.pre_clip_gradient_norm_mean
    )
    elapsed_ms = None if telemetry is None else telemetry.elapsed_ms
    fields.extend([
        f"selection={selection_text}",
        f"loss={result.loss:.6f}",
        f"mean_mae={observed('mean_return_mae')}",
        f"sigma_mae={observed('sigma_return_mae')}",
        f"tp_mae={observed('prob_tp_mae')}",
        f"sl_mae={observed('prob_sl_mae')}",
        f"vol_mae={observed('volatility_next_mae')}",
        f"hit_mae={observed('hitting_prob_tp_mae')}",
        (
            "grad_mean=n/a"
            if gradient_mean is None
            else f"grad_mean={gradient_mean:.3f}"
        ),
        f"rows={result.rows}",
        f"batches={result.batches}",
        "time=n/a" if elapsed_ms is None else f"time={elapsed_ms / 1000.0:.1f}s",
        f"stage={result.loss_stage}/{max_loss_stage}",
    ])
    return " ".join(fields)


__all__ = [
    "EpochTelemetry",
    "ObservedTrainingEpoch",
    "TargetErrorObservation",
    "epoch_telemetry_document",
    "format_epoch_console_line",
    "format_epoch_log_line",
]
