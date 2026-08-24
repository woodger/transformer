from __future__ import annotations

from typing import Protocol

from app.service.domain.authentication import AuthenticatedPrincipal


class InvalidAccessTokenError(Exception):
    """The supplied credential is not an active API access token."""


class AccessTokenAuthenticator(Protocol):
    def authenticate(self, access_token: str) -> AuthenticatedPrincipal: ...


__all__ = [
    "AccessTokenAuthenticator",
    "InvalidAccessTokenError",
]
