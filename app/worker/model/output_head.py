import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHead(nn.Module):
    """Produce raw public coordinates and private objective resources."""

    def __init__(
        self,
        hidden_dim: int,
        target_width: int,
        resource_classes: tuple[str, ...],
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if target_width <= 0:
            raise ValueError("target_width must be a positive integer")
        if any(
            resource_class != "PositiveScalarPerObservation"
            for resource_class in resource_classes
        ):
            raise ValueError("private resource class is unavailable")

        self.hidden_dim = hidden_dim
        self.target_width = target_width
        self.resource_classes = resource_classes
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.target_head = nn.Linear(128, target_width)
        self.resource_heads = nn.ModuleList(
            nn.Linear(128, 1) for _resource_class in resource_classes
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _shared = self.forward_with_shared_representation(hidden_states)
        return output

    def forward_with_shared_representation(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_dim:
            raise ValueError("hidden states must have shape [batch, hidden]")

        shared = self.shared(hidden_states)
        targets = self.target_head(shared)
        resources = [
            F.softplus(head(shared)) + 1e-6
            for head in self.resource_heads
        ]
        if resources:
            targets = torch.cat((targets, *resources), dim=1)
        return targets, shared


__all__ = ["OutputHead"]
