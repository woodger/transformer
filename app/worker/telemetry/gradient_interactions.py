from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from app.contracts.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class GradientInteractionObservation:
    """One sampled set of objective gradients at the shared head input."""

    component_names: tuple[str, ...]
    pair_names: tuple[tuple[str, str], ...]
    values: tuple[torch.Tensor, ...]

    @classmethod
    def evaluate(
        cls,
        components: tuple[tuple[str, torch.Tensor], ...],
        shared_representation: torch.Tensor,
    ) -> GradientInteractionObservation:
        names = tuple(name for name, _loss in components)
        gradients = tuple(
            torch.autograd.grad(
                loss,
                shared_representation,
                retain_graph=True,
                create_graph=False,
            )[0].detach().float().reshape(-1)
            for _name, loss in components
        )
        norms: tuple[torch.Tensor, ...] = tuple(
            torch.sqrt(torch.sum(gradient * gradient))
            for gradient in gradients
        )
        pairs: list[tuple[str, str]] = []
        cosines: list[torch.Tensor] = []
        for left_index, left in enumerate(gradients):
            for right_index in range(left_index + 1, len(gradients)):
                right = gradients[right_index]
                denominator = norms[left_index] * norms[right_index]
                cosine = torch.where(
                    denominator > 0,
                    torch.dot(left, right) / denominator,
                    denominator.new_tensor(float("nan")),
                )
                pairs.append((names[left_index], names[right_index]))
                cosines.append(cosine)
        return cls(
            component_names=names,
            pair_names=tuple(pairs),
            values=(*norms, *cosines),
        )

    def materialize(self) -> JsonObject:
        host = cast(
            list[float],
            torch.stack(self.values).cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
        )
        component_count = len(self.component_names)
        return {
            "components": [
                {"name": name, "norm": float(host[index])}
                for index, name in enumerate(self.component_names)
            ],
            "pairs": [
                {
                    "left": left,
                    "right": right,
                    "cosine": float(host[component_count + index]),
                }
                for index, (left, right) in enumerate(self.pair_names)
            ],
        }


__all__ = ["GradientInteractionObservation"]
