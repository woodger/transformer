from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """Stable owner identity established at the service boundary."""

    owner_subject: str


def is_valid_owner_subject(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 256
        and not any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        )
    )


__all__ = ["AuthenticatedPrincipal", "is_valid_owner_subject"]
