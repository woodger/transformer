from __future__ import annotations

from typing import Protocol

from app.service.domain.authentication import AuthenticatedPrincipal


class InvalidAccessTokenError(Exception):
    """The supplied credential is not an active OAuth access token."""


class InsufficientAccessError(Exception):
    """The active token lacks the required audience or scope."""


class AuthenticationUnavailableError(Exception):
    """The external authentication authority cannot make a safe decision."""


class AccessTokenAuthenticator(Protocol):
    def authenticate(self, access_token: str) -> AuthenticatedPrincipal: ...


__all__ = [
    "AccessTokenAuthenticator",
    "AuthenticationUnavailableError",
    "InsufficientAccessError",
    "InvalidAccessTokenError",
]
