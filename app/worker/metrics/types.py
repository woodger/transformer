import math
from dataclasses import dataclass

_LOSS_FIELDS = tuple(f"loss_l{index}" for index in range(6))
_SEMANTICS = (
    "mean_return",
    "sigma_return",
    "prob_tp",
    "prob_sl",
    "volatility_next",
    "hitting_prob_tp",
)


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
    grad_norm: float = 0.0
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

    def update(
        self,
        rows: int,
        loss_parts: dict[str, float],
        grad_norm: float,
        nan_ratio: float,
        masked_token_ratio: float = 0.0,
        complete_token_ratio: float = 0.0,
        partial_token_ratio: float = 0.0,
        empty_token_ratio: float = 0.0,
    ):
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

        self.grad_norm = average(self.grad_norm, grad_norm)
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

    def direct_losses(self) -> tuple[float, ...]:
        if self.rows <= 0:
            raise ValueError("selection score is incomplete: epoch has no rows")
        values = tuple(getattr(self, name) for name in _LOSS_FIELDS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("selection score contains a non-finite component")
        return values

    def log_line(self, **extra) -> str:
        fields = {
            **extra,
            "loss": f"{self.loss:.6f}",
            **{
                name: f"{getattr(self, name):.6f}"
                for name in _LOSS_FIELDS
            },
            "nll": f"{self.loss_nll:.6f}",
            "ev": f"{self.loss_ev:.6f}",
            "grad": f"{self.grad_norm:.3f}",
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

    def console_line(self, **extra) -> str:
        fields = []
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
        fields.extend([
            f"selection={selection_text}",
            f"loss={self.loss:.6f}",
            f"mean_mae={self.mean_return_mae:.6g}",
            f"sigma_mae={self.sigma_return_mae:.6g}",
            f"tp_mae={self.prob_tp_mae:.6g}",
            f"sl_mae={self.prob_sl_mae:.6g}",
            f"vol_mae={self.volatility_next_mae:.6g}",
            f"hit_mae={self.hitting_prob_tp_mae:.6g}",
            f"grad={self.grad_norm:.3f}",
            f"rows={self.rows}",
            f"batches={self.batches}",
            f"time={self.elapsed_ms / 1000.0:.1f}s",
            f"stage={self.loss_stage}/{max_loss_stage}",
        ])
        return " ".join(fields)

    def to_dict(self, **extra) -> dict:
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
            "grad_norm": self.grad_norm,
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

    def __gt__(self, other) -> bool:
        return self.loss > float(other)
