from __future__ import annotations

import hmac
import http.client
import json
import secrets
import ssl
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
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
from app.service.domain.authentication import (
    AuthenticatedPrincipal,
    is_valid_owner_subject,
)

_MAX_RESPONSE_BYTES = 64 * 1024
_ACCESS_TOKEN_TYPE = "bearer"


class _AuthorizationDecision(Enum):
    ALLOW = auto()
    INVALID = auto()
    DENIED = auto()
    UNAVAILABLE = auto()


@dataclass(frozen=True, slots=True)
class _AuthorizationResult:
    decision: _AuthorizationDecision
    principal: AuthenticatedPrincipal | None = None
    expires_at: int | None = None


@dataclass(frozen=True, slots=True)
class _AuthorizationCacheEntry:
    result: _AuthorizationResult
    deadline: float


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
        self._cache_secret = secrets.token_bytes(32)
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[bytes, _AuthorizationCacheEntry] = (
            OrderedDict()
        )
        self._inflight: dict[bytes, Future[_AuthorizationResult]] = {}

    def authenticate(self, access_token: str) -> AuthenticatedPrincipal:
        cache_key = hmac.digest(
            self._cache_secret,
            access_token.encode("utf-8"),
            "sha256",
        )
        result = self._cached_result(cache_key)
        if result is None:
            result = self._authorize_singleflight(cache_key, access_token)
        return _resolve_authorization(result)

    def _authorize_singleflight(
        self,
        cache_key: bytes,
        access_token: str,
    ) -> _AuthorizationResult:
        with self._cache_lock:
            cached = self._cached_result_locked(
                cache_key,
                monotonic_now=time.monotonic(),
                wall_now=time.time(),
            )
            if cached is not None:
                return cached
            pending = self._inflight.get(cache_key)
            if pending is None:
                pending = Future[_AuthorizationResult]()
                self._inflight[cache_key] = pending
                is_leader = True
            else:
                is_leader = False

        if not is_leader:
            return pending.result()

        try:
            result = self._authorize_uncached(access_token)
        except BaseException as exc:
            with self._cache_lock:
                pending.set_exception(exc)
                self._inflight.pop(cache_key, None)
            raise

        with self._cache_lock:
            self._store_result_locked(cache_key, result)
            pending.set_result(result)
            self._inflight.pop(cache_key, None)
        return result

    def _authorize_uncached(self, access_token: str) -> _AuthorizationResult:
        try:
            document = self._introspect(access_token)
            return self._authorization_result(document)
        except AuthenticationUnavailableError:
            return _AuthorizationResult(_AuthorizationDecision.UNAVAILABLE)

    def _authorization_result(
        self,
        document: JsonObject,
    ) -> _AuthorizationResult:
        active = document.get("active")
        if not isinstance(active, bool):
            raise AuthenticationUnavailableError(
                "Hydra introspection response is incomplete"
            )
        if not active:
            return _AuthorizationResult(_AuthorizationDecision.INVALID)

        token_type = document.get("token_type")
        if not isinstance(token_type, str) or not token_type:
            raise AuthenticationUnavailableError(
                "Hydra introspection response has no token type"
            )
        if token_type.casefold() != _ACCESS_TOKEN_TYPE:
            return _AuthorizationResult(_AuthorizationDecision.INVALID)

        client_id = document.get("client_id")
        if not is_valid_owner_subject(client_id):
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
            return _AuthorizationResult(_AuthorizationDecision.DENIED)

        expires_at = document.get("exp")
        if expires_at is not None:
            if isinstance(expires_at, bool) or not isinstance(expires_at, int):
                raise AuthenticationUnavailableError(
                    "Hydra introspection response has an invalid expiry"
                )
            if expires_at <= int(time.time()):
                return _AuthorizationResult(_AuthorizationDecision.INVALID)

        return _AuthorizationResult(
            decision=_AuthorizationDecision.ALLOW,
            principal=AuthenticatedPrincipal(owner_subject=client_id),
            expires_at=expires_at,
        )

    def _cached_result(
        self,
        cache_key: bytes,
    ) -> _AuthorizationResult | None:
        with self._cache_lock:
            return self._cached_result_locked(
                cache_key,
                monotonic_now=time.monotonic(),
                wall_now=time.time(),
            )

    def _cached_result_locked(
        self,
        cache_key: bytes,
        *,
        monotonic_now: float,
        wall_now: float,
    ) -> _AuthorizationResult | None:
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        expires_at = entry.result.expires_at
        if (
            monotonic_now >= entry.deadline
            or expires_at is not None
            and wall_now >= expires_at
        ):
            del self._cache[cache_key]
            return None
        self._cache.move_to_end(cache_key)
        return entry.result

    def _store_result_locked(
        self,
        cache_key: bytes,
        result: _AuthorizationResult,
    ) -> None:
        cache_config = self.config.authorization_cache
        if result.decision is _AuthorizationDecision.ALLOW:
            ttl_seconds = cache_config.positive_ttl_seconds
            if result.expires_at is not None:
                ttl_seconds = min(
                    ttl_seconds,
                    result.expires_at - time.time(),
                )
        elif result.decision in {
            _AuthorizationDecision.INVALID,
            _AuthorizationDecision.DENIED,
        }:
            ttl_seconds = cache_config.negative_ttl_seconds
        else:
            return
        if ttl_seconds <= 0:
            return
        self._cache[cache_key] = _AuthorizationCacheEntry(
            result=result,
            deadline=time.monotonic() + ttl_seconds,
        )
        self._cache.move_to_end(cache_key)
        while len(self._cache) > cache_config.max_entries:
            self._cache.popitem(last=False)

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


def _resolve_authorization(
    result: _AuthorizationResult,
) -> AuthenticatedPrincipal:
    if result.decision is _AuthorizationDecision.ALLOW:
        if result.principal is None:
            raise AuthenticationUnavailableError(
                "cached authorization result is incomplete"
            )
        return result.principal
    if result.decision is _AuthorizationDecision.INVALID:
        raise InvalidAccessTokenError("access token is inactive")
    if result.decision is _AuthorizationDecision.DENIED:
        raise InsufficientAccessError(
            "access token lacks the required audience or scope"
        )
    raise AuthenticationUnavailableError(
        "Hydra introspection is unavailable"
    )


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


__all__ = ["HydraAccessTokenAuthenticator"]
