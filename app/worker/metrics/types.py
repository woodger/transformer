import math
from dataclasses import dataclass, field

from app.contracts.json_types import JsonObject, JsonValue

_LOSS_FIELDS = tuple(f"loss_l{index}" for index in range(6))
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


@dataclass
class TrainMetrics:
    rows: int = 0
    batches: int = 0
    loss: float = 0.0
    loss_l0: float = 0.0
    loss_l1: float = 0.0
    loss_l2: float = 0.0
    loss_l3: float = 0.0
    loss_l4: float = 0.0
    loss_l5: float = 0.0
    loss_nll: float = 0.0
    loss_ev: float = 0.0
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
    selection_score: float | None = None
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
    step: int = 0
    lr: float = 0.0
    loss_stage: int = 0
    minimum_loss_stage: int = 0
    maximum_loss_stage: int = 0
    _finite_gradient_norms: list[float] = field(
        default_factory=_empty_float_list,
        repr=False,
    )

    def update(
        self,
        rows: int,
        loss_parts: dict[str, float],
        grad_norm: float,
        optimizer_update_applied: bool,
        amp_overflow: bool,
        nan_ratio: float,
        masked_token_ratio: float = 0.0,
        complete_token_ratio: float = 0.0,
        partial_token_ratio: float = 0.0,
        empty_token_ratio: float = 0.0,
    ) -> None:
        total_rows = self.rows + rows
        if total_rows <= 0:
            return

        def average(current: float, value: float) -> float:
            return math.fsum((current * self.rows, value * rows)) / total_rows

        for name in (
            "loss",
            *_LOSS_FIELDS,
            "loss_nll",
            "loss_ev",
            *(f"{semantic}_mae" for semantic in _SEMANTICS),
        ):
            setattr(self, name, average(getattr(self, name), loss_parts[name]))

        for semantic in _SEMANTICS:
            rmse_name = f"{semantic}_rmse"
            mse_name = f"{semantic}_mse"
            mse = math.fsum((
                getattr(self, rmse_name) ** 2 * self.rows,
                loss_parts[mse_name] * rows,
            )) / total_rows
            setattr(self, rmse_name, math.sqrt(max(0.0, mse)))

        gradient_is_finite = math.isfinite(grad_norm)
        if gradient_is_finite and grad_norm < 0:
            raise ValueError("pre-clip gradient norm must not be negative")
        if amp_overflow and (optimizer_update_applied or gradient_is_finite):
            raise ValueError(
                "AMP overflow requires a skipped update and non-finite gradient"
            )
        self.training_batches_completed += 1
        if optimizer_update_applied:
            self.optimizer_updates_applied += 1
        else:
            self.optimizer_updates_skipped += 1
        if amp_overflow:
            self.amp_overflow_batches += 1
        if gradient_is_finite:
            self.finite_gradient_batches += 1
            self._finite_gradient_norms.append(grad_norm)
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
        self.rows = total_rows
        self.batches += 1
        self.step = int(loss_parts.get("step", self.step))
        observed_stage = int(loss_parts.get("loss_stage", self.loss_stage))
        self.loss_stage = observed_stage
        if self.minimum_loss_stage == 0:
            self.minimum_loss_stage = observed_stage
        else:
            self.minimum_loss_stage = min(
                self.minimum_loss_stage,
                observed_stage,
            )
        self.maximum_loss_stage = max(self.maximum_loss_stage, observed_stage)

    def finalize_gradient_statistics(self) -> None:
        if self._finite_gradient_norms:
            ordered = sorted(self._finite_gradient_norms)
            self.pre_clip_gradient_norm_mean = (
                math.fsum(ordered) / len(ordered)
            )
            self.pre_clip_gradient_norm_max = ordered[-1]
            nearest_rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
            self.pre_clip_gradient_norm_p95 = ordered[nearest_rank]
        elif self.finite_gradient_batches == 0:
            self.pre_clip_gradient_norm_mean = None
            self.pre_clip_gradient_norm_max = None
            self.pre_clip_gradient_norm_p95 = None
        self._validate_gradient_telemetry()

    def _validate_gradient_telemetry(self) -> None:
        if self.training_batches_completed != self.batches:
            raise ValueError(
                "training batch telemetry differs from the epoch batch count"
            )
        if self.training_batches_completed != (
            self.optimizer_updates_applied + self.optimizer_updates_skipped
        ):
            raise ValueError("optimizer update telemetry is inconsistent")
        if self.training_batches_completed != (
            self.finite_gradient_batches + self.non_finite_gradient_batches
        ):
            raise ValueError("gradient batch telemetry is inconsistent")
        if (
            self.amp_overflow_batches > self.optimizer_updates_skipped
            or self.amp_overflow_batches > self.non_finite_gradient_batches
        ):
            raise ValueError("AMP overflow telemetry is inconsistent")
        statistics = (
            self.pre_clip_gradient_norm_mean,
            self.pre_clip_gradient_norm_max,
            self.pre_clip_gradient_norm_p95,
        )
        if self.finite_gradient_batches == 0:
            if any(value is not None for value in statistics):
                raise ValueError(
                    "gradient statistics require at least one finite batch"
                )
            return
        if any(
            value is None or not math.isfinite(value) or value < 0
            for value in statistics
        ):
            raise ValueError("finite gradient statistics are incomplete")
        mean, maximum, percentile = statistics
        if mean is None or maximum is None or percentile is None:
            raise AssertionError("gradient statistics were not narrowed")
        if mean > maximum or percentile > maximum:
            raise ValueError("gradient statistics are inconsistent")

    def direct_losses(self) -> tuple[float, ...]:
        if self.rows <= 0:
            raise ValueError("selection score is incomplete: epoch has no rows")
        values = tuple(getattr(self, name) for name in _LOSS_FIELDS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("selection score contains a non-finite component")
        return values

    def log_line(self, **extra: object) -> str:
        self.finalize_gradient_statistics()
        gradient_mean = self.pre_clip_gradient_norm_mean
        fields: dict[str, object] = {
            **extra,
            "loss": f"{self.loss:.6f}",
            **{
                name: f"{getattr(self, name):.6f}"
                for name in _LOSS_FIELDS
            },
            "nll": f"{self.loss_nll:.6f}",
            "ev": f"{self.loss_ev:.6f}",
            "grad_mean": (
                "n/a" if gradient_mean is None else f"{gradient_mean:.3f}"
            ),
            "rows": self.rows,
            "batches": self.batches,
            "nan": f"{self.nan_ratio:.4f}",
            "masked_tokens": f"{self.masked_token_ratio:.4f}",
            "complete_tokens": f"{self.complete_token_ratio:.4f}",
            "partial_tokens": f"{self.partial_token_ratio:.4f}",
            "empty_tokens": f"{self.empty_token_ratio:.4f}",
            "step": self.step,
            "lr": f"{self.lr:.6g}",
            "loss_stage": self.loss_stage,
            "minimum_loss_stage": self.minimum_loss_stage,
            "maximum_loss_stage": self.maximum_loss_stage,
            "ms": f"{self.elapsed_ms:.0f}",
        }
        return " ".join(f"{key}={value}" for key, value in fields.items())

    def console_line(self, **extra: object) -> str:
        self.finalize_gradient_statistics()
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
        max_loss_stage = extra.get("max_loss_stage", self.loss_stage)
        gradient_mean = self.pre_clip_gradient_norm_mean
        fields.extend([
            f"selection={selection_text}",
            f"loss={self.loss:.6f}",
            f"mean_mae={self.mean_return_mae:.6g}",
            f"sigma_mae={self.sigma_return_mae:.6g}",
            f"tp_mae={self.prob_tp_mae:.6g}",
            f"sl_mae={self.prob_sl_mae:.6g}",
            f"vol_mae={self.volatility_next_mae:.6g}",
            f"hit_mae={self.hitting_prob_tp_mae:.6g}",
            (
                "grad_mean=n/a"
                if gradient_mean is None
                else f"grad_mean={gradient_mean:.3f}"
            ),
            f"rows={self.rows}",
            f"batches={self.batches}",
            f"time={self.elapsed_ms / 1000.0:.1f}s",
            f"stage={self.loss_stage}/{max_loss_stage}",
        ])
        return " ".join(fields)

    def to_dict(self, **extra: JsonValue) -> JsonObject:
        self.finalize_gradient_statistics()
        return {
            **extra,
            "rows": self.rows,
            "batches": self.batches,
            "loss": self.loss,
            **{name: getattr(self, name) for name in _LOSS_FIELDS},
            "loss_nll": self.loss_nll,
            "loss_ev": self.loss_ev,
            **{
                f"{semantic}_mae": getattr(self, f"{semantic}_mae")
                for semantic in _SEMANTICS
            },
            **{
                f"{semantic}_rmse": getattr(self, f"{semantic}_rmse")
                for semantic in _SEMANTICS
            },
            "selection_score": self.selection_score,
            "trainingBatchesCompleted": self.training_batches_completed,
            "optimizerUpdatesApplied": self.optimizer_updates_applied,
            "optimizerUpdatesSkipped": self.optimizer_updates_skipped,
            "ampOverflowBatches": self.amp_overflow_batches,
            "finiteGradientBatches": self.finite_gradient_batches,
            "nonFiniteGradientBatches": self.non_finite_gradient_batches,
            "preClipGradientNormMean": self.pre_clip_gradient_norm_mean,
            "preClipGradientNormMax": self.pre_clip_gradient_norm_max,
            "preClipGradientNormP95": self.pre_clip_gradient_norm_p95,
            "nan_ratio": self.nan_ratio,
            "masked_token_ratio": self.masked_token_ratio,
            "complete_token_ratio": self.complete_token_ratio,
            "partial_token_ratio": self.partial_token_ratio,
            "empty_token_ratio": self.empty_token_ratio,
            "input_pipeline_ms": self.input_pipeline_ms,
            "missing_stats_ms": self.missing_stats_ms,
            "host_to_device_ms": self.host_to_device_ms,
            "train_step_ms": self.train_step_ms,
            "elapsed_ms": self.elapsed_ms,
            "step": self.step,
            "lr": self.lr,
            "loss_stage": self.loss_stage,
            "minimum_loss_stage": self.minimum_loss_stage,
            "maximum_loss_stage": self.maximum_loss_stage,
        }

    def __float__(self) -> float:
        return self.loss

    def __format__(self, format_spec: str) -> str:
        return format(self.loss, format_spec)

    def __gt__(self, other: float | int) -> bool:
        return self.loss > float(other)
