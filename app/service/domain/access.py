from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AccessTokenRecord:
    token_id: str
    subject: str
    created_at: datetime
    revoked_at: datetime | None
    token: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class AuthIdentity:
    token_id: str
    subject: str


__all__ = ["AccessTokenRecord", "AuthIdentity"]
