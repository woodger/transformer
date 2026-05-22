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
):
    loss_stage = validate_loss_stage(loss_stage)
    components = active_loss_components(loss_stage)

    meanR, sigmaR, logitTP, logitSL, volNext, logitHit = preds.T
    t_meanR, _, _, _, t_volNext, t_hitTP = targets.T

    loss = preds.new_tensor(0.0)
    loss_prob = preds.new_tensor(0.0)
    loss_ev = preds.new_tensor(0.0)
    loss_vol = preds.new_tensor(0.0)

    # -------------------------
    # Gaussian NLL
    # -------------------------
    var = sigmaR ** 2 + 1e-6
    loss_ret = torch.mean(
        (t_meanR - meanR) ** 2 / (2 * var) + torch.log(sigmaR)
    )
    loss += loss_ret

    # -------------------------
    # Probabilities (AMP safe)
    # -------------------------
    if "prob" in components:
        raw_loss_prob = (
            F.binary_cross_entropy_with_logits(logitTP, t_hitTP) +
            F.binary_cross_entropy_with_logits(logitSL, 1 - t_hitTP)
        )
        loss_prob = 0.5 * raw_loss_prob
        loss += loss_prob

    # -------------------------
    # Bayesian EV
    # -------------------------
    if "ev" in components:
        pTP = torch.sigmoid(logitTP)
        pSL = torch.sigmoid(logitSL)

        ev = pTP - pSL
        risk_pen = sigmaR.detach() * torch.abs(ev)
        loss_ev = -0.3 * torch.mean(ev - 0.1 * risk_pen)
        loss += loss_ev

    # -------------------------
    # Volatility
    # -------------------------
    if "vol" in components:
        raw_loss_vol = torch.mean(
            (torch.log(volNext + 1e-6) - torch.log(t_volNext + 1e-6)) ** 2
        )
        loss_vol = 0.2 * raw_loss_vol
        loss += loss_vol

    if not return_parts:
        return loss

    sigma_values = sigmaR.detach().float().reshape(-1)
    ret_error = (meanR.detach().float() - t_meanR.detach().float()).reshape(-1)
    ret_abs_error = torch.abs(ret_error)
    ret_squared_error = ret_error ** 2
    ret_baseline_abs_error = torch.abs(t_meanR.detach().float()).reshape(-1)

    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_ret": float(loss_ret.detach().cpu()),
        "loss_prob": float(loss_prob.detach().cpu()),
        "loss_ev": float(loss_ev.detach().cpu()),
        "loss_vol": float(loss_vol.detach().cpu()),
        "sigma_min": float(torch.min(sigma_values).cpu()),
        "sigma_p05": float(torch.quantile(sigma_values, 0.05).cpu()),
        "sigma_mean": float(torch.mean(sigma_values).cpu()),
        "ret_mae": float(torch.mean(ret_abs_error).cpu()),
        "ret_mse": float(torch.mean(ret_squared_error).cpu()),
        "ret_mae_baseline": float(torch.mean(ret_baseline_abs_error).cpu()),
        "loss_stage": loss_stage,
    }
