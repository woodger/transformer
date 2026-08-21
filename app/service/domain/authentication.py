from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """Stable owner identity established at the service boundary."""

    owner_subject: str


__all__ = ["AuthenticatedPrincipal"]
