from collections.abc import Iterable

import torch


def parameter_tree_stats(
    parameters: Iterable[torch.Tensor],
) -> dict[str, float]:
    cpu_tensors = [
        parameter.detach().cpu().flatten()
        for parameter in parameters
    ]
    flattened = torch.cat(cpu_tensors)
    return {
        "mean": float(flattened.mean()),
        "std": float(flattened.std()),
        "norm": float(flattened.square().sum().sqrt()),
    }


__all__ = ["parameter_tree_stats"]
