from __future__ import annotations

from dataclasses import dataclass

import torch

from app.contracts.semantic.v1 import ModelContract
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
        model_contract: ModelContract,
    ) -> TargetErrorObservation:
        selected = model_contract.target_identities
        predictions = public_predictions(model_output, model_contract).detach().float()
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
