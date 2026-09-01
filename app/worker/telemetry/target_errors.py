from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from app.contracts.ml import canonical_targets
from app.worker.model.transformer import public_predictions


@dataclass(frozen=True, slots=True)
class TargetErrorObservation:
    """Device-resident target errors awaiting one batch scalar transfer."""

    targets: tuple[str, ...]
    values: tuple[torch.Tensor, ...]

    @classmethod
    def evaluate(
        cls,
        model_output: torch.Tensor,
        expected: torch.Tensor,
        targets: Sequence[str],
        *,
        include_return_scale: bool,
    ) -> TargetErrorObservation:
        selected = canonical_targets(targets)
        predictions = public_predictions(
            model_output,
            selected,
            include_return_scale=include_return_scale,
        ).detach().float()
        errors = predictions - expected.detach().float()
        absolute_errors = errors.abs()
        squared_errors = errors.square()
        return cls(
            targets=selected,
            values=(
                *(absolute_errors[:, index].mean() for index in range(len(selected))),
                *(squared_errors[:, index].mean() for index in range(len(selected))),
            ),
        )

    def decode(
        self,
        values: tuple[float, ...],
    ) -> dict[str, tuple[float, float]]:
        width = len(self.targets)
        if len(values) != width * 2:
            raise ValueError("target error observation has an invalid width")
        return {
            target: (values[index], values[index + width])
            for index, target in enumerate(self.targets)
        }


__all__ = ["TargetErrorObservation"]
