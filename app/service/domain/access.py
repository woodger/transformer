from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime

ACCESS_TOKEN_TTL_MONTHS = 3


@dataclass(frozen=True, slots=True)
class AccessTokenRecord:
    token_id: str
    subject: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    token: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class AuthIdentity:
    token_id: str
    subject: str
    expires_at: datetime


def access_token_expiration(created_at: datetime) -> datetime:
    month_index = created_at.month - 1 + ACCESS_TOKEN_TTL_MONTHS
    year = created_at.year + month_index // 12
    month = month_index % 12 + 1
    day = min(created_at.day, monthrange(year, month)[1])
    return created_at.replace(year=year, month=month, day=day)


__all__ = [
    "ACCESS_TOKEN_TTL_MONTHS",
    "AccessTokenRecord",
    "AuthIdentity",
    "access_token_expiration",
]
