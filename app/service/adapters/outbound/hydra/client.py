from __future__ import annotations

import http.client
import json
import ssl
import time
from typing import cast
from urllib.parse import urlencode, urlsplit

from app.contracts.json_types import JsonObject
from app.service.adapters.outbound.hydra.config import (
    INTROSPECTION_PATH,
    INTROSPECTION_TIMEOUT_SECONDS,
    REQUIRED_AUDIENCE,
    REQUIRED_SCOPE,
    HydraIntrospectionConfig,
)
from app.service.application.ports.authentication import (
    AuthenticationUnavailableError,
    InsufficientAccessError,
    InvalidAccessTokenError,
)
from app.service.domain.authentication import AuthenticatedPrincipal

_MAX_RESPONSE_BYTES = 64 * 1024
_ACCESS_TOKEN_TYPE = "bearer"


class HydraAccessTokenAuthenticator:
    """Authenticate one opaque access token through Hydra introspection."""

    def __init__(self, config: HydraIntrospectionConfig) -> None:
        self.config = config
        parsed = urlsplit(config.endpoint)
        if parsed.hostname is None:
            raise ValueError("Hydra introspection endpoint has no hostname")
        self._host = parsed.hostname
        self._use_tls = parsed.scheme == "https"
        self._port = parsed.port or (443 if self._use_tls else 80)
        self._path = INTROSPECTION_PATH
        self._tls_context = (
            ssl.create_default_context() if self._use_tls else None
        )

    def authenticate(self, access_token: str) -> AuthenticatedPrincipal:
        document = self._introspect(access_token)
        active = document.get("active")
        if not isinstance(active, bool):
            raise AuthenticationUnavailableError(
                "Hydra introspection response is incomplete"
            )
        if not active:
            raise InvalidAccessTokenError("access token is inactive")

        token_type = document.get("token_type")
        if not isinstance(token_type, str) or not token_type:
            raise AuthenticationUnavailableError(
                "Hydra introspection response has no token type"
            )
        if token_type.casefold() != _ACCESS_TOKEN_TYPE:
            raise InvalidAccessTokenError("credential is not an access token")

        client_id = document.get("client_id")
        if not _valid_owner_subject(client_id):
            raise AuthenticationUnavailableError(
                "Hydra introspection response has no valid client ID"
            )

        audiences = _audiences(document.get("aud"))
        scope = document.get("scope")
        if scope is None:
            scopes: set[str] = set()
        elif isinstance(scope, str):
            scopes = set(scope.split(" "))
        else:
            raise AuthenticationUnavailableError(
                "Hydra introspection response has an invalid scope"
            )
        if REQUIRED_AUDIENCE not in audiences or REQUIRED_SCOPE not in scopes:
            raise InsufficientAccessError(
                "access token lacks the required audience or scope"
            )

        expires_at = document.get("exp")
        if expires_at is not None:
            if isinstance(expires_at, bool) or not isinstance(expires_at, int):
                raise AuthenticationUnavailableError(
                    "Hydra introspection response has an invalid expiry"
                )
            if expires_at <= int(time.time()):
                raise InvalidAccessTokenError("access token is expired")

        return AuthenticatedPrincipal(owner_subject=cast(str, client_id))

    def _introspect(self, access_token: str) -> JsonObject:
        body = urlencode({"token": access_token}).encode("ascii")
        connection: http.client.HTTPConnection
        if self._use_tls:
            connection = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=INTROSPECTION_TIMEOUT_SECONDS,
                context=self._tls_context,
            )
        else:
            connection = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=INTROSPECTION_TIMEOUT_SECONDS,
            )
        try:
            connection.connect()
            if connection.sock is not None:
                connection.sock.settimeout(INTROSPECTION_TIMEOUT_SECONDS)
            connection.request(
                "POST",
                self._path,
                body=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            response = connection.getresponse()
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
            status = response.status
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise AuthenticationUnavailableError(
                "Hydra introspection is unavailable"
            ) from exc
        finally:
            connection.close()
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise AuthenticationUnavailableError(
                "Hydra introspection response is too large"
            )
        if not 200 <= status < 300:
            raise AuthenticationUnavailableError(
                "Hydra introspection returned a non-success status"
            )
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthenticationUnavailableError(
                "Hydra introspection response is not valid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise AuthenticationUnavailableError(
                "Hydra introspection response is not an object"
            )
        return cast(JsonObject, decoded)


def _audiences(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        items = cast(list[object], value)
        if all(isinstance(item, str) for item in items):
            return set(cast(list[str], items))
    raise AuthenticationUnavailableError(
        "Hydra introspection response has an invalid audience"
    )


def _valid_owner_subject(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 256
        and not any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        )
    )


__all__ = ["HydraAccessTokenAuthenticator"]
