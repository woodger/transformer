from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

ACCESS_TOKEN_TTL_DAYS = 180


@dataclass(frozen=True, slots=True)
class AccessTokenRecord:
    token_id: str
    subject: str
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime | None = None
    token: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class AuthIdentity:
    token_id: str
    subject: str
    expires_at: datetime


def access_token_expiration(created_at: datetime) -> datetime:
    return created_at + timedelta(days=ACCESS_TOKEN_TTL_DAYS)


__all__ = [
    "ACCESS_TOKEN_TTL_DAYS",
    "AccessTokenRecord",
    "AuthIdentity",
    "access_token_expiration",
]
