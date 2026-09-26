from __future__ import annotations

from app.service.application.ports.authentication import InvalidAccessTokenError
from app.service.domain.authentication import AuthenticatedPrincipal


class StaticAccessTokenAuthenticator:
    """Детерминированный authenticator границы сервиса для изолированных tests."""

    def __init__(self, credentials: dict[str, str]) -> None:
        self._credentials = credentials
        self.calls: list[str] = []

    def authenticate(self, access_token: str) -> AuthenticatedPrincipal:
        self.calls.append(access_token)
        owner_subject = self._credentials.get(access_token)
        if owner_subject is None:
            raise InvalidAccessTokenError("inactive test credential")
        return AuthenticatedPrincipal(owner_subject=owner_subject)


__all__ = ["StaticAccessTokenAuthenticator"]
