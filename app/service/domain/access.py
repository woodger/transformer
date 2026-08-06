from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AccessTokenRecord:
    token_id: str
    subject: str
    created_at: datetime
    revoked_at: datetime | None
    token: str | None = None


@dataclass(frozen=True)
class AuthIdentity:
    token_id: str
    subject: str


__all__ = ["AccessTokenRecord", "AuthIdentity"]
