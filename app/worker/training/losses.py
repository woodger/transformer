from dataclasses import dataclass

import torch
import torch.nn.functional as F

from app.config import LOSS_SCHEDULE, LOSS_STAGE, STAGE_SIZE


@dataclass(frozen=True)
class LossStageDefinition:
    stage: int
    name: str
    components: tuple[str, ...]


LOSS_STAGE_DEFINITIONS = (
    LossStageDefinition(1, "returns", ("ret",)),
    LossStageDefinition(2, "probabilities", ("ret", "prob")),
    LossStageDefinition(3, "bayesian-ev", ("ret", "prob", "ev")),
    LossStageDefinition(4, "volatility", ("ret", "prob", "ev", "vol")),
)

LOSS_STAGES = len(LOSS_STAGE_DEFINITIONS)
LOSS_SCHEDULES = ("none", "epoch", "step")
_LOSS_STATISTIC_NAMES = (
    "loss",
    "loss_ret",
    "loss_prob",
    "loss_ev",
    "loss_vol",
    "sigma_min",
    "sigma_p05",
    "sigma_mean",
    "ret_mae",
    "ret_mse",
    "ret_mae_baseline",
)


@dataclass(frozen=True)
class LossStatistics:
    """Device-resident scalar statistics awaiting one host transfer."""

    values: tuple[torch.Tensor, ...]
    loss_stage: int

    def materialize(
        self,
        grad_norm: torch.Tensor | None = None,
    ) -> dict[str, float | int] | tuple[dict[str, float | int], float]:
        device_values = self.values
        if grad_norm is not None:
            device_values = (*device_values, grad_norm.detach())
        host_values = torch.stack(tuple(
            value.detach().reshape(())
            for value in device_values
        )).cpu().tolist()
        parts = dict(zip(
            _LOSS_STATISTIC_NAMES,
            host_values[:len(_LOSS_STATISTIC_NAMES)],
            strict=True,
        ))
        parts["loss_stage"] = self.loss_stage
        if grad_norm is None:
            return parts
        return parts, host_values[-1]


def validate_loss_stage(loss_stage: int) -> int:
    if loss_stage < 1 or loss_stage > LOSS_STAGES:
        raise ValueError(f"loss_stage must be between 1 and {LOSS_STAGES}")
    return loss_stage


def validate_stage_size(stage_size: int) -> int:
    if stage_size <= 0:
        raise ValueError("stage_size must be a positive integer")
    return stage_size


def validate_loss_schedule(loss_schedule: str) -> str:
    if loss_schedule not in LOSS_SCHEDULES:
        choices = ", ".join(LOSS_SCHEDULES)
        raise ValueError(f"loss_schedule must be one of: {choices}")
    return loss_schedule


def resolve_loss_stage(
    progress: int,
    loss_schedule: str = LOSS_SCHEDULE,
    stage_size: int = STAGE_SIZE,
    max_stage: int = LOSS_STAGE,
) -> int:
    max_stage = validate_loss_stage(max_stage)
    loss_schedule = validate_loss_schedule(loss_schedule)

    if loss_schedule == "none":
        return max_stage

    stage_size = validate_stage_size(stage_size)
    return min(max_stage, progress // stage_size + 1)


def active_loss_components(loss_stage: int) -> tuple[str, ...]:
    loss_stage = validate_loss_stage(loss_stage)
    return LOSS_STAGE_DEFINITIONS[loss_stage - 1].components


def combined_loss(
    preds,
    targets,
    loss_stage: int = LOSS_STAGE,
    return_parts: bool = False,
    return_statistics: bool = False,
):
    if return_parts and return_statistics:
        raise ValueError(
            "return_parts and return_statistics are mutually exclusive"
        )
    loss_stage = validate_loss_stage(loss_stage)
    components = active_loss_components(loss_stage)

    (
        mean_return,
        return_scale,
        take_profit_logit,
        stop_loss_logit,
        next_volatility,
        _hit_logit,
    ) = preds.T
    target_mean_return, _, _, _, target_next_volatility, target_hit = targets.T

    loss = preds.new_tensor(0.0)
    loss_prob = preds.new_tensor(0.0)
    loss_ev = preds.new_tensor(0.0)
    loss_vol = preds.new_tensor(0.0)

    # -------------------------
    # Gaussian NLL
    # -------------------------
    var = return_scale.square() + 1e-6
    loss_ret = torch.mean(
        0.5
        * (
            (target_mean_return - mean_return).square() / var
            + torch.log(var)
        )
    )
    loss += loss_ret

    # -------------------------
    # Probabilities (AMP safe)
    # -------------------------
    if "prob" in components:
        raw_loss_prob = (
            F.binary_cross_entropy_with_logits(take_profit_logit, target_hit)
            + F.binary_cross_entropy_with_logits(
                stop_loss_logit,
                1 - target_hit,
            )
        )
        loss_prob = 0.5 * raw_loss_prob
        loss += loss_prob

    # -------------------------
    # Bayesian EV
    # -------------------------
    if "ev" in components:
        take_profit_probability = torch.sigmoid(take_profit_logit)
        stop_loss_probability = torch.sigmoid(stop_loss_logit)

        ev = take_profit_probability - stop_loss_probability
        risk_pen = return_scale.detach() * torch.abs(ev)
        loss_ev = -0.3 * torch.mean(ev - 0.1 * risk_pen)
        loss += loss_ev

    # -------------------------
    # Volatility
    # -------------------------
    if "vol" in components:
        raw_loss_vol = torch.mean(
            (
                torch.log(next_volatility + 1e-6)
                - torch.log(target_next_volatility + 1e-6)
            )
            ** 2
        )
        loss_vol = 0.2 * raw_loss_vol
        loss += loss_vol

    if not return_parts and not return_statistics:
        return loss

    sigma_values = return_scale.detach().float().reshape(-1)
    ret_error = (
        mean_return.detach().float()
        - target_mean_return.detach().float()
    ).reshape(-1)
    ret_abs_error = torch.abs(ret_error)
    ret_squared_error = ret_error ** 2
    ret_baseline_abs_error = torch.abs(
        target_mean_return.detach().float()
    ).reshape(-1)

    statistics = LossStatistics(
        values=(
            loss.detach(),
            loss_ret.detach(),
            loss_prob.detach(),
            loss_ev.detach(),
            loss_vol.detach(),
            torch.min(sigma_values),
            torch.quantile(sigma_values, 0.05),
            torch.mean(sigma_values),
            torch.mean(ret_abs_error),
            torch.mean(ret_squared_error),
            torch.mean(ret_baseline_abs_error),
        ),
        loss_stage=loss_stage,
    )
    if return_statistics:
        return loss, statistics
    return loss, statistics.materialize()
