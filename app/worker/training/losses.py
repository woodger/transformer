from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from app.contracts.worker.v9.objective import ObjectiveConfig
from app.worker.model.transformer import public_predictions


@dataclass(frozen=True, slots=True)
class MaterializedLossStatistics:
    loss: float
    direct_losses: tuple[float, ...]
    auxiliary_losses: tuple[tuple[str, float], ...]
    grad_norm: float | None
    observations: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class LossStatistics:
    loss: torch.Tensor
    direct_losses: tuple[torch.Tensor, ...]
    auxiliary_losses: tuple[tuple[str, torch.Tensor], ...]

    def materialize(
        self,
        grad_norm: torch.Tensor | None = None,
        observations: tuple[torch.Tensor, ...] = (),
    ) -> MaterializedLossStatistics:
        core_values = (
            self.loss,
            *self.direct_losses,
            *(value for _operator, value in self.auxiliary_losses),
        )
        device_values = (*core_values, *observations)
        if grad_norm is not None:
            device_values = (*device_values, grad_norm.detach())
        host_values = cast(
            list[float],
            torch.stack(tuple(
                value.detach().reshape(())
                for value in device_values
            )).cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
        )
        core_count = len(core_values)
        direct_count = len(self.direct_losses)
        auxiliary_count = len(self.auxiliary_losses)
        return MaterializedLossStatistics(
            loss=float(host_values[0]),
            direct_losses=tuple(
                float(value)
                for value in host_values[1:1 + direct_count]
            ),
            auxiliary_losses=tuple(
                (operator, float(value))
                for (operator, _tensor), value in zip(
                    self.auxiliary_losses,
                    host_values[
                        1 + direct_count:
                        1 + direct_count + auxiliary_count
                    ],
                    strict=True,
                )
            ),
            grad_norm=(
                None if grad_norm is None else float(host_values[-1])
            ),
            observations=tuple(
                float(value)
                for value in host_values[core_count:core_count + len(observations)]
            ),
        )


@dataclass(frozen=True, slots=True)
class LossEvaluation:
    loss: torch.Tensor
    statistics: LossStatistics
    diagnostic_components: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True, slots=True)
class MaterializedLossEvaluation:
    loss: torch.Tensor
    statistics: MaterializedLossStatistics


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    objective: ObjectiveConfig,
    *,
    return_parts: Literal[False] = False,
    return_statistics: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    objective: ObjectiveConfig,
    *,
    return_parts: Literal[True],
    return_statistics: Literal[False] = False,
) -> MaterializedLossEvaluation: ...


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    objective: ObjectiveConfig,
    *,
    return_parts: Literal[False] = False,
    return_statistics: Literal[True],
) -> LossEvaluation: ...


def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    objective: ObjectiveConfig,
    *,
    return_parts: bool = False,
    return_statistics: bool = False,
) -> torch.Tensor | MaterializedLossEvaluation | LossEvaluation:
    """Evaluate one immutable declarative objective for every optimizer step."""

    if return_parts and return_statistics:
        raise ValueError(
            "return_parts and return_statistics are mutually exclusive"
        )
    expected_output_width = (
        objective.target_width + int(objective.requires_return_scale)
    )
    if model_output.ndim != 2 or model_output.shape[1] != expected_output_width:
        raise ValueError(
            f"model output must have shape [rows, {expected_output_width}]"
        )
    if targets.ndim != 2 or targets.shape != (
        model_output.shape[0],
        objective.target_width,
    ):
        raise ValueError(
            f"targets must have shape [rows, {objective.target_width}]"
        )

    direct_means: list[torch.Tensor] = []
    task_components: dict[str, torch.Tensor] = {}
    loss = model_output.new_tensor(0.0)
    target_indices = {
        target: index
        for index, target in enumerate(objective.targets)
    }

    for index, specification in enumerate(objective.direct_losses):
        target = str(specification["target"])
        operator = str(specification["operator"])
        direct = _direct_loss(
            operator,
            model_output[:, index],
            targets[:, index],
        ).mean()
        weighted = _number(specification["weight"], "direct loss weight") * direct
        direct_means.append(direct)
        task_components[f"target:{target}"] = weighted
        loss = loss + weighted

    auxiliary_means: list[tuple[str, torch.Tensor]] = []
    return_scale = (
        model_output[:, objective.target_width]
        if objective.requires_return_scale
        else None
    )
    predictions = public_predictions(
        model_output,
        objective.targets,
        include_return_scale=objective.requires_return_scale,
    )

    for specification in objective.auxiliary_losses:
        operator = str(specification["operator"])
        auxiliary = _auxiliary_loss(
            operator,
            specification,
            model_output,
            predictions,
            targets,
            target_indices,
            return_scale,
        ).mean()
        weighted = _number(
            specification["weight"],
            "auxiliary loss weight",
        ) * auxiliary
        auxiliary_means.append((operator, auxiliary))
        loss = loss + weighted
        component_name = f"auxiliary:{operator}"
        if operator == "GaussianNLL":
            component_name = "target:MeanReturn"
        task_components[component_name] = (
            task_components.get(component_name, model_output.new_tensor(0.0))
            + weighted
        )

    if not return_parts and not return_statistics:
        return loss

    statistics = LossStatistics(
        loss=loss.detach(),
        direct_losses=tuple(value.detach() for value in direct_means),
        auxiliary_losses=tuple(
            (operator, value.detach())
            for operator, value in auxiliary_means
        ),
    )
    if return_statistics:
        return LossEvaluation(
            loss=loss,
            statistics=statistics,
            diagnostic_components=tuple(task_components.items()),
        )
    return MaterializedLossEvaluation(
        loss=loss,
        statistics=statistics.materialize(),
    )


def _direct_loss(
    operator: str,
    model_value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    if operator == "SmoothL1":
        return F.smooth_l1_loss(model_value, target_value, reduction="none")
    if operator == "BinaryCrossEntropyWithLogits":
        return F.binary_cross_entropy_with_logits(
            model_value,
            target_value,
            reduction="none",
        )
    if operator == "LogMSE":
        return (
            torch.log(model_value + 1e-6)
            - torch.log(target_value + 1e-6)
        ).square()
    raise ValueError(f"unsupported direct loss operator: {operator}")


def _auxiliary_loss(
    operator: str,
    specification: Mapping[str, object],
    model_output: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_indices: dict[str, int],
    return_scale: torch.Tensor | None,
) -> torch.Tensor:
    if operator == "GaussianNLL":
        if return_scale is None:
            raise ValueError("GaussianNLL requires returnScale")
        index = target_indices["MeanReturn"]
        variance = return_scale.square() + 1e-6
        return 0.5 * (
            (targets[:, index] - model_output[:, index]).square() / variance
            + torch.log(variance)
        )

    probability_delta = (
        predictions[:, target_indices["ProbTP"]]
        - predictions[:, target_indices["ProbSL"]]
    )
    if operator == "ExpectedValue":
        return -probability_delta
    if operator == "RiskAdjustedExpectedValue":
        if return_scale is None:
            raise ValueError("RiskAdjustedExpectedValue requires returnScale")
        risk_penalty = return_scale.detach() * torch.abs(probability_delta)
        return -(
            probability_delta
            - _number(
                specification["riskPenalty"],
                "risk adjustment penalty",
            ) * risk_penalty
        )
    raise ValueError(f"unsupported auxiliary loss operator: {operator}")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


__all__ = [
    "LossEvaluation",
    "LossStatistics",
    "MaterializedLossEvaluation",
    "MaterializedLossStatistics",
    "combined_loss",
]
