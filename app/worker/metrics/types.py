import math
from dataclasses import dataclass


@dataclass
class TrainMetrics:
    rows: int = 0
    batches: int = 0
    loss: float = 0.0
    loss_ret: float = 0.0
    loss_prob: float = 0.0
    loss_ev: float = 0.0
    loss_vol: float = 0.0
    sigma_min: float = 0.0
    sigma_p05: float = 0.0
    sigma_mean: float = 0.0
    ret_mae: float = 0.0
    ret_rmse: float = 0.0
    ret_mae_baseline: float = 0.0
    ret_mae_skill: float = 0.0
    ret_mae_improvement: float = 0.0
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

        def avg(current: float, value: float) -> float:
            return ((current * self.rows) + (value * rows)) / total_rows

        self.loss = avg(self.loss, loss_parts["loss"])
        self.loss_ret = avg(self.loss_ret, loss_parts["loss_ret"])
        self.loss_prob = avg(self.loss_prob, loss_parts["loss_prob"])
        self.loss_ev = avg(self.loss_ev, loss_parts["loss_ev"])
        self.loss_vol = avg(self.loss_vol, loss_parts["loss_vol"])
        self.sigma_min = avg(self.sigma_min, loss_parts.get("sigma_min", 0.0))
        self.sigma_p05 = avg(self.sigma_p05, loss_parts.get("sigma_p05", 0.0))
        self.sigma_mean = avg(self.sigma_mean, loss_parts.get("sigma_mean", 0.0))
        self.ret_mae = avg(self.ret_mae, loss_parts.get("ret_mae", 0.0))
        ret_mse = (
            (self.ret_rmse ** 2) * self.rows
            + loss_parts.get("ret_mse", 0.0) * rows
        ) / total_rows
        self.ret_rmse = math.sqrt(ret_mse)
        self.ret_mae_baseline = avg(
            self.ret_mae_baseline,
            loss_parts.get("ret_mae_baseline", 0.0),
        )
        if self.ret_mae_baseline > 0:
            self.ret_mae_skill = self.ret_mae / self.ret_mae_baseline
            self.ret_mae_improvement = 1.0 - self.ret_mae_skill
        else:
            self.ret_mae_skill = math.inf
            self.ret_mae_improvement = -math.inf
        self.grad_norm = avg(self.grad_norm, grad_norm)
        self.nan_ratio = avg(self.nan_ratio, nan_ratio)
        self.masked_token_ratio = avg(self.masked_token_ratio, masked_token_ratio)
        self.complete_token_ratio = avg(self.complete_token_ratio, complete_token_ratio)
        self.partial_token_ratio = avg(self.partial_token_ratio, partial_token_ratio)
        self.empty_token_ratio = avg(self.empty_token_ratio, empty_token_ratio)
        self.rows = total_rows
        self.batches += 1
        self.step = int(loss_parts.get("step", self.step))
        self.loss_stage = int(loss_parts.get("loss_stage", self.loss_stage))

    def log_line(self, **extra) -> str:
        fields = {
            **extra,
            "loss": f"{self.loss:.6f}",
            "ret": f"{self.loss_ret:.6f}",
            "prob": f"{self.loss_prob:.6f}",
            "ev": f"{self.loss_ev:.6f}",
            "vol": f"{self.loss_vol:.6f}",
            "sigma_min": f"{self.sigma_min:.6g}",
            "sigma_p05": f"{self.sigma_p05:.6g}",
            "sigma_mean": f"{self.sigma_mean:.6g}",
            "ret_mae": f"{self.ret_mae:.6g}",
            "ret_rmse": f"{self.ret_rmse:.6g}",
            "ret_mae_baseline": f"{self.ret_mae_baseline:.6g}",
            "ret_mae_skill": f"{self.ret_mae_skill:.6g}",
            "ret_mae_improvement": f"{self.ret_mae_improvement:.6g}",
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
            "ms": f"{self.elapsed_ms:.0f}",
        }

        return " ".join(f"{key}={value}" for key, value in fields.items())

    def console_line(self, **extra) -> str:
        fields = []
        for key in ("frame", "epoch"):
            if key in extra and extra[key] is not None:
                fields.append(f"{key}={extra[key]}")

        monitor_value = extra.get("monitor_value")
        if isinstance(monitor_value, (int, float)) and math.isfinite(monitor_value):
            monitor = f"{monitor_value:.6g}"
        else:
            monitor = "n/a"

        if self.ret_mae_baseline > 0.0 and math.isfinite(self.ret_mae_skill):
            skill = f"{self.ret_mae_skill:.6g}x"
            status = "BETTER" if self.ret_mae_skill < 1.0 else "WORSE"
        else:
            skill = "n/a"
            status = "N/A"

        max_loss_stage = extra.get("max_loss_stage", self.loss_stage)
        fields.extend([
            f"monitor_value={monitor}",
            f"loss={self.loss:.6f}",
            f"mae={self.ret_mae:.6g}",
            f"baseline={self.ret_mae_baseline:.6g}",
            f"skill={skill}",
            f"status={status}",
            f"sigma={self.sigma_mean:.6g}",
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
            "loss_ret": self.loss_ret,
            "loss_prob": self.loss_prob,
            "loss_ev": self.loss_ev,
            "loss_vol": self.loss_vol,
            "sigma_min": self.sigma_min,
            "sigma_p05": self.sigma_p05,
            "sigma_mean": self.sigma_mean,
            "ret_mae": self.ret_mae,
            "ret_rmse": self.ret_rmse,
            "ret_mae_baseline": self.ret_mae_baseline,
            "ret_mae_skill": self.ret_mae_skill,
            "ret_mae_improvement": self.ret_mae_improvement,
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
        }

    def __float__(self) -> float:
        return self.loss

    def __format__(self, format_spec: str) -> str:
        return format(self.loss, format_spec)

    def __gt__(self, other) -> bool:
        return self.loss > float(other)
