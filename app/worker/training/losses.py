from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from app.contracts.worker.v4.config import (
    DEFAULT_DIRECT_LOSS_WEIGHTS,
    DEFAULT_LOSS_SCHEDULE,
    DEFAULT_LOSS_STAGE,
    DEFAULT_STAGE_SIZE,
)
from app.worker.model.transformer import public_predictions


@dataclass(frozen=True, slots=True)
class LossStageDefinition:
    stage: int
    name: str
    components: tuple[str, ...]


LOSS_STAGE_DEFINITIONS = (
    LossStageDefinition(1, "returns", ("L0", "L1", "nll")),
    LossStageDefinition(
        2,
        "probabilities",
        ("L0", "L1", "L2", "L3", "L5", "nll"),
    ),
    LossStageDefinition(
        3,
        "bayesian-ev",
        ("L0", "L1", "L2", "L3", "L5", "nll", "ev"),
    ),
    LossStageDefinition(
        4,
        "volatility",
        ("L0", "L1", "L2", "L3", "L4", "L5", "nll", "ev"),
    ),
)

LOSS_STAGES = len(LOSS_STAGE_DEFINITIONS)
LOSS_SCHEDULES = ("none", "epoch", "step")
_SEMANTIC_NAMES = (
    "mean_return",
    "sigma_return",
    "prob_tp",
    "prob_sl",
    "volatility_next",
    "hitting_prob_tp",
)
_LOSS_STATISTIC_NAMES = (
    "loss",
    *(f"loss_l{index}" for index in range(6)),
    "loss_nll",
    "loss_ev",
    *(f"{name}_mae" for name in _SEMANTIC_NAMES),
    *(f"{name}_mse" for name in _SEMANTIC_NAMES),
)


@dataclass(frozen=True, slots=True)
class MaterializedLossStatistics:
    """Host-side scalar metrics produced by one synchronized transfer."""

    parts: dict[str, float | int]
    grad_norm: float | None


@dataclass(frozen=True, slots=True)
class LossStatistics:
    """Device-resident row-normalized statistics awaiting one host transfer."""

    values: tuple[torch.Tensor, ...]
    loss_stage: int

    def materialize(
        self,
        grad_norm: torch.Tensor | None = None,
    ) -> MaterializedLossStatistics:
        device_values = self.values
        if grad_norm is not None:
            device_values = (*device_values, grad_norm.detach())
        # PyTorch types ``Tensor.tolist`` as a list of unknown depth. The
        # stacked tensor is one-dimensional by construction here.
        host_values = cast(
            list[float],
            torch.stack(tuple(
                value.detach().reshape(())
                for value in device_values
            )).cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
        )
        parts: dict[str, float | int] = dict(zip(
            _LOSS_STATISTIC_NAMES,
            host_values[:len(_LOSS_STATISTIC_NAMES)],
            strict=True,
        ))
        parts["loss_stage"] = self.loss_stage
        return MaterializedLossStatistics(
            parts=parts,
            grad_norm=None if grad_norm is None else host_values[-1],
        )


@dataclass(frozen=True, slots=True)
class LossEvaluation:
    """Differentiable loss paired with device-resident statistics."""

    loss: torch.Tensor
    statistics: LossStatistics


@dataclass(frozen=True, slots=True)
class MaterializedLossEvaluation:
    """Differentiable loss paired with host-side scalar statistics."""

    loss: torch.Tensor
    statistics: MaterializedLossStatistics


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
    loss_schedule: str = DEFAULT_LOSS_SCHEDULE,
    stage_size: int = DEFAULT_STAGE_SIZE,
    max_stage: int = DEFAULT_LOSS_STAGE,
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


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    loss_stage: int = DEFAULT_LOSS_STAGE,
    direct_loss_weights: tuple[float, ...] = DEFAULT_DIRECT_LOSS_WEIGHTS,
    return_parts: Literal[False] = False,
    return_statistics: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    loss_stage: int,
    direct_loss_weights: tuple[float, ...],
    return_parts: Literal[True],
    return_statistics: Literal[False] = False,
) -> MaterializedLossEvaluation: ...


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    loss_stage: int,
    direct_loss_weights: tuple[float, ...],
    return_parts: Literal[False] = False,
    return_statistics: Literal[True] = True,
) -> LossEvaluation: ...


def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    loss_stage: int = DEFAULT_LOSS_STAGE,
    direct_loss_weights: tuple[float, ...] = DEFAULT_DIRECT_LOSS_WEIGHTS,
    return_parts: bool = False,
    return_statistics: bool = False,
) -> (
    torch.Tensor
    | MaterializedLossEvaluation
    | LossEvaluation
):
    """Evaluate the target-aligned staged objective.

    The first six model heads represent public target coordinates. Probability
    heads remain logits here for stable BCE; the seventh head is the private
    Gaussian return scale. Every public coordinate has its own direct loss.
    """

    if return_parts and return_statistics:
        raise ValueError(
            "return_parts and return_statistics are mutually exclusive"
        )
    if model_output.ndim != 2 or model_output.shape[1] != 7:
        raise ValueError("model output must have shape [rows, 7]")
    if targets.ndim != 2 or targets.shape != (model_output.shape[0], 6):
        raise ValueError("targets must have shape [rows, 6]")
    if len(direct_loss_weights) != 6:
        raise ValueError("direct_loss_weights must contain six values")

    loss_stage = validate_loss_stage(loss_stage)
    active = frozenset(active_loss_components(loss_stage))
    mean_return = model_output[:, 0]
    sigma_return = model_output[:, 1]
    take_profit_logit = model_output[:, 2]
    stop_loss_logit = model_output[:, 3]
    next_volatility = model_output[:, 4]
    hitting_probability_logit = model_output[:, 5]
    return_scale = model_output[:, 6]

    direct_rows = (
        F.smooth_l1_loss(mean_return, targets[:, 0], reduction="none"),
        F.smooth_l1_loss(sigma_return, targets[:, 1], reduction="none"),
        F.binary_cross_entropy_with_logits(
            take_profit_logit,
            targets[:, 2],
            reduction="none",
        ),
        F.binary_cross_entropy_with_logits(
            stop_loss_logit,
            targets[:, 3],
            reduction="none",
        ),
        (
            torch.log(next_volatility + 1e-6)
            - torch.log(targets[:, 4] + 1e-6)
        ).square(),
        F.binary_cross_entropy_with_logits(
            hitting_probability_logit,
            targets[:, 5],
            reduction="none",
        ),
    )
    direct_means = tuple(values.mean() for values in direct_rows)

    loss = model_output.new_tensor(0.0)
    for index, direct in enumerate(direct_means):
        if f"L{index}" in active:
            loss = loss + float(direct_loss_weights[index]) * direct

    variance = return_scale.square() + 1e-6
    nll_rows = 0.5 * (
        (targets[:, 0] - mean_return).square() / variance
        + torch.log(variance)
    )
    loss_nll = nll_rows.mean()
    if "nll" in active:
        loss = loss + loss_nll

    predictions = public_predictions(model_output)
    ev = predictions[:, 2] - predictions[:, 3]
    risk_penalty = return_scale.detach() * torch.abs(ev)
    loss_ev = -0.3 * torch.mean(ev - 0.1 * risk_penalty)
    if "ev" in active:
        loss = loss + loss_ev

    if not return_parts and not return_statistics:
        return loss

    errors = predictions.detach().float() - targets.detach().float()
    absolute_errors = errors.abs()
    squared_errors = errors.square()
    statistics = LossStatistics(
        values=(
            loss.detach(),
            *(value.detach() for value in direct_means),
            loss_nll.detach(),
            loss_ev.detach(),
            *(absolute_errors[:, index].mean() for index in range(6)),
            *(squared_errors[:, index].mean() for index in range(6)),
        ),
        loss_stage=loss_stage,
    )
    if return_statistics:
        return LossEvaluation(loss=loss, statistics=statistics)
    return MaterializedLossEvaluation(
        loss=loss,
        statistics=statistics.materialize(),
    )
