import hashlib
import re
import time

import pyarrow.flight as flight

from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.token_cache import AuthIdentity

MIDDLEWARE_KEY = "auth"
_BEARER_TOKEN = re.compile(r"^[\x21-\x7e]{1,4096}$")


class BearerAuthMiddleware(flight.ServerMiddleware):
    def __init__(
        self,
        subject: str,
        method: str,
        metrics: OperationalMetrics,
        logger: JsonLogger,
    ):
        self.subject = subject
        self.method = method
        self._metrics = metrics
        self._logger = logger
        self._started = time.monotonic()

    def call_completed(self, exception):
        elapsed = time.monotonic() - self._started
        status = "OK" if exception is None else type(exception).__name__
        self._metrics.record_rpc(self.method, status, elapsed)
        self._logger.event(
            "flight.rpc.completed",
            method=self.method,
            status=status,
            latencyMs=round(elapsed * 1000.0, 3),
        )


class BearerAuthMiddlewareFactory(flight.ServerMiddlewareFactory):
    def __init__(
        self,
        token_cache,
        metrics: OperationalMetrics,
        logger: JsonLogger,
    ):
        if isinstance(token_cache, dict):
            token_cache = InMemoryAccessTokenCache(token_cache)
        self._token_cache = token_cache
        self._metrics = metrics
        self._logger = logger

    def start_call(self, info, headers):
        started = time.monotonic()
        method = str(getattr(info, "method", "unknown"))
        value = _single_authorization_header(headers)
        if value is None:
            self._reject(method, started, "missing bearer authorization metadata")
        scheme, separator, token = value.partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or not _BEARER_TOKEN.fullmatch(token)
        ):
            self._reject(method, started, "invalid bearer authorization metadata")

        identity = self._token_cache.lookup(token)
        if identity is None:
            self._reject(method, started, "invalid bearer credential")

        return BearerAuthMiddleware(
            subject=identity.subject,
            method=method,
            metrics=self._metrics,
            logger=self._logger,
        )

    def _reject(self, method: str, started: float, message: str) -> None:
        elapsed = time.monotonic() - started
        self._metrics.record_rpc(method, "UNAUTHENTICATED", elapsed)
        self._logger.event(
            "flight.rpc.completed",
            method=method,
            status="UNAUTHENTICATED",
            latencyMs=round(elapsed * 1000.0, 3),
        )
        # Never include the supplied authorization value in logs or errors.
        raise flight.FlightUnauthenticatedError(f"UNAUTHENTICATED: {message}")


def authenticated_subject(context) -> str:
    middleware = context.get_middleware(MIDDLEWARE_KEY)
    if middleware is None or not getattr(middleware, "subject", None):
        raise flight.FlightUnauthenticatedError(
            "UNAUTHENTICATED: authentication middleware is unavailable"
        )
    return middleware.subject


def _single_authorization_header(headers) -> str | None:
    values = headers.get("authorization")
    if not values:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) != 1:
        return None
    value = values[0]
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    return value if isinstance(value, str) else None


def _validate_credential(token, subject) -> None:
    if not isinstance(token, str) or not _BEARER_TOKEN.fullmatch(token):
        raise ValueError(
            "bearer token must contain 1-4096 non-whitespace printable ASCII characters"
        )
    if (
        not isinstance(subject, str)
        or not subject
        or len(subject) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in subject)
    ):
        raise ValueError(
            "bearer subject must be 1-256 characters without control characters"
        )


def validate_bearer_credentials(tokens: dict[str, str]) -> dict[str, str]:
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("at least one bearer token is required")
    for token, subject in tokens.items():
        _validate_credential(token, subject)
    return dict(tokens)


class InMemoryAccessTokenCache:
    """Small injection seam used by isolated middleware/server tests."""

    def __init__(self, tokens: dict[str, str]):
        credentials = validate_bearer_credentials(tokens)
        self._entries = {
            hashlib.sha256(token.encode("ascii")).hexdigest(): AuthIdentity(
                token_id="in-memory",
                subject=subject,
            )
            for token, subject in credentials.items()
        }

    def lookup(self, token: str) -> AuthIdentity | None:
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        return self._entries.get(digest)
