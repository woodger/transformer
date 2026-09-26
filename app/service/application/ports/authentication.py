from __future__ import annotations

from typing import Protocol

from app.service.domain.authentication import AuthenticatedPrincipal


class InvalidAccessTokenError(Exception):
    """Переданные учётные данные не являются активным токеном доступа API."""


class AccessTokenAuthenticator(Protocol):
    def authenticate(self, access_token: str) -> AuthenticatedPrincipal: ...


__all__ = [
    "AccessTokenAuthenticator",
    "InvalidAccessTokenError",
]
