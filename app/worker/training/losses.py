from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from app.contracts.semantic.v6 import ModelContract
from app.worker.model.transformer import apply_transformation, public_predictions


@dataclass(frozen=True, slots=True)
class MaterializedLossStatistics:
    loss: float
    direct_losses: tuple[float, ...]
    auxiliary_losses: tuple[tuple[str, str, float], ...]
    grad_norm: float | None
    observations: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class LossStatistics:
    loss: torch.Tensor
    direct_losses: tuple[torch.Tensor, ...]
    auxiliary_losses: tuple[tuple[str, str, torch.Tensor], ...]

    def materialize(
        self,
        grad_norm: torch.Tensor | None = None,
        observations: tuple[torch.Tensor, ...] = (),
    ) -> MaterializedLossStatistics:
        core_values = (
            self.loss,
            *self.direct_losses,
            *(value for _identity, _operator, value in self.auxiliary_losses),
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
        loss_index = 0
        direct_start = loss_index + 1
        direct_end = direct_start + len(self.direct_losses)
        auxiliary_start = direct_end
        auxiliary_end = auxiliary_start + len(self.auxiliary_losses)
        observations_start = auxiliary_end
        observations_end = observations_start + len(observations)

        loss_value = host_values[loss_index]
        direct_loss_values = host_values[direct_start:direct_end]
        auxiliary_loss_values = host_values[auxiliary_start:auxiliary_end]
        observation_values = host_values[observations_start:observations_end]
        grad_norm_value = (
            None if grad_norm is None else host_values[observations_end]
        )
        return MaterializedLossStatistics(
            loss=float(loss_value),
            direct_losses=tuple(
                float(value)
                for value in direct_loss_values
            ),
            auxiliary_losses=tuple(
                (identity, operator, float(value))
                for (identity, operator, _tensor), value in zip(
                    self.auxiliary_losses,
                    auxiliary_loss_values,
                    strict=True,
                )
            ),
            grad_norm=None if grad_norm_value is None else float(grad_norm_value),
            observations=tuple(
                float(value)
                for value in observation_values
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
    model_contract: ModelContract,
    *,
    return_parts: Literal[False] = False,
    return_statistics: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    model_contract: ModelContract,
    *,
    return_parts: Literal[True],
    return_statistics: Literal[False] = False,
) -> MaterializedLossEvaluation: ...


@overload
def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    model_contract: ModelContract,
    *,
    return_parts: Literal[False] = False,
    return_statistics: Literal[True],
) -> LossEvaluation: ...


def combined_loss(
    model_output: torch.Tensor,
    targets: torch.Tensor,
    model_contract: ModelContract,
    *,
    return_parts: bool = False,
    return_statistics: bool = False,
) -> torch.Tensor | MaterializedLossEvaluation | LossEvaluation:
    """Вычислить один неизменяемый декларативный objective на каждом шаге оптимизатора."""

    if return_parts and return_statistics:
        raise ValueError(
            "return_parts and return_statistics are mutually exclusive"
        )
    expected_output_width = (
        model_contract.target_width
        + len(model_contract.resource_declarations)
    )
    if model_output.ndim != 2 or model_output.shape[1] != expected_output_width:
        raise ValueError(
            f"model output must have shape [rows, {expected_output_width}]"
        )
    if targets.ndim != 2 or targets.shape != (
        model_output.shape[0],
        model_contract.target_width,
    ):
        raise ValueError(
            f"targets must have shape [rows, {model_contract.target_width}]"
        )

    direct_means: list[torch.Tensor] = []
    task_components: dict[str, torch.Tensor] = {}
    loss = model_output.new_tensor(0.0)
    target_indices = {
        identity: index
        for index, identity in enumerate(model_contract.target_identities)
    }
    resource_indices = {
        str(resource["identity"]): model_contract.target_width + index
        for index, resource in enumerate(model_contract.resource_declarations)
    }

    for index, specification in enumerate(model_contract.direct_components):
        operator = str(specification["operator"])
        slot = model_contract.target_slots[index]
        estimate = apply_transformation(
            model_output[:, index],
            cast(str, slot["lossInputTransformation"]),
        )
        direct = _direct_loss(
            operator,
            specification,
            estimate,
            targets[:, index],
        ).mean()
        weighted = _number(specification["weight"], "direct loss weight") * direct
        direct_means.append(direct)
        task_components[str(specification["identity"])] = weighted
        loss = loss + weighted

    auxiliary_means: list[tuple[str, str, torch.Tensor]] = []
    predictions = public_predictions(model_output, model_contract)

    for specification in model_contract.auxiliary_components:
        component_identity = str(specification["identity"])
        operator = str(specification["operator"])
        auxiliary = _auxiliary_loss(
            operator,
            specification,
            model_output,
            predictions,
            targets,
            target_indices,
            resource_indices,
            model_contract,
        ).mean()
        weighted = _number(
            specification["weight"],
            "auxiliary loss weight",
        ) * auxiliary
        auxiliary_means.append((component_identity, operator, auxiliary))
        loss = loss + weighted
        task_components[component_identity] = weighted

    if not return_parts and not return_statistics:
        return loss

    statistics = LossStatistics(
        loss=loss.detach(),
        direct_losses=tuple(value.detach() for value in direct_means),
        auxiliary_losses=tuple(
            (identity, operator, value.detach())
            for identity, operator, value in auxiliary_means
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
    specification: Mapping[str, object],
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
    if operator == "PositiveClassWeightedBinaryCrossEntropyWithLogits":
        if model_value.dtype in (torch.float16, torch.bfloat16):
            model_value = model_value.float()
        positive_class_weight = model_value.new_tensor(
            _number(
                specification["positiveClassWeight"],
                "positive class weight",
            )
        )
        return F.binary_cross_entropy_with_logits(
            model_value,
            target_value,
            reduction="none",
            pos_weight=positive_class_weight,
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
    resource_indices: dict[str, int],
    model_contract: ModelContract,
) -> torch.Tensor:
    roles = cast(Mapping[str, object], specification["roles"])
    if operator == "BernoulliConfidencePenalty":
        index = _target_index(roles["probability"], target_indices)
        logit = model_output[:, index]
        # Для AMP повышаем точность коррекции logit и расчёта энтропии,
        # особенно при вероятностях около границ [0, 1].
        if logit.dtype in (torch.float16, torch.bfloat16):
            logit = logit.float()
        positive_class_weight = model_contract.positive_class_weight_for_target(
            index,
        )
        # Энтропия относится к публичной p = sigmoid(z - ln(positiveClassWeight)).
        # Без коррекции weighted BCE регуляризировал бы другую вероятность.
        if positive_class_weight is not None:
            logit = logit - math.log(positive_class_weight)

        # Возвращаем -H(p): положительный weight поощряет рост энтропии.
        # logsigmoid(±logit) избегает log(0) при насыщении sigmoid.
        return (
            torch.sigmoid(logit) * F.logsigmoid(logit)
            + torch.sigmoid(-logit) * F.logsigmoid(-logit)
        )

    if operator == "GaussianNLL":
        index = _target_index(roles["locationEstimate"], target_indices)
        scale = model_output[
            :,
            _resource_index(roles["scale"], resource_indices),
        ]
        estimate = apply_transformation(
            model_output[:, index],
            cast(
                str,
                model_contract.target_slots[index]["lossInputTransformation"],
            ),
        )
        variance = scale.square() + 1e-6
        return 0.5 * (
            (targets[:, index] - estimate).square() / variance
            + torch.log(variance)
        )

    positive_index = _target_index(
        roles["positiveOutcomeProbability"],
        target_indices,
    )
    negative_index = _target_index(
        roles["negativeOutcomeProbability"],
        target_indices,
    )
    probability_delta = (
        predictions[:, positive_index]
        - predictions[:, negative_index]
    )
    if operator == "ExpectedValue":
        return -probability_delta
    if operator == "RiskAdjustedExpectedValue":
        scale = model_output[
            :,
            _resource_index(roles["uncertaintyScale"], resource_indices),
        ]
        risk_penalty = scale.detach() * torch.abs(probability_delta)
        return -(
            probability_delta
            - _number(
                cast(Mapping[str, object], specification["parameters"])[
                    "riskPenalty"
                ],
                "risk adjustment penalty",
            ) * risk_penalty
        )
    raise ValueError(f"unsupported auxiliary loss operator: {operator}")


def _target_index(value: object, indices: Mapping[str, int]) -> int:
    if not isinstance(value, str):
        raise ValueError("target role must be an opaque target identity")
    return indices[value]


def _resource_index(value: object, indices: Mapping[str, int]) -> int:
    if not isinstance(value, str):
        raise ValueError("resource role must be an opaque resource identity")
    return indices[value]


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
