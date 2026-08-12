import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from typing import NoReturn, Protocol, cast

import pyarrow.flight as flight

from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.domain.access import AuthIdentity

MIDDLEWARE_KEY = "auth"
_BEARER_TOKEN = re.compile(r"^[\x21-\x7e]{1,4096}$")


class _TokenCache(Protocol):
    def lookup(self, token: str) -> AuthIdentity | None: ...


class _CallContext(Protocol):
    def get_middleware(self, key: str) -> object | None: ...


class _FlightExceptionFactory(Protocol):
    def __call__(self, message: str) -> Exception: ...


class BearerAuthMiddleware(
    flight.ServerMiddleware,  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage, reportUntypedBaseClass]
):
    def __init__(
        self,
        subject: str,
        method: str,
        metrics: OperationalMetrics,
        logger: JsonLogger,
    ) -> None:
        self.subject = subject
        self.method = method
        self._metrics = metrics
        self._logger = logger
        self._started = time.monotonic()

    def call_completed(self, exception: BaseException | None) -> None:
        elapsed = time.monotonic() - self._started
        status = "OK" if exception is None else type(exception).__name__
        self._metrics.record_rpc(self.method, status, elapsed)
        self._logger.event(
            "flight.rpc.completed",
            method=self.method,
            status=status,
            latencyMs=round(elapsed * 1000.0, 3),
        )


class BearerAuthMiddlewareFactory(
    flight.ServerMiddlewareFactory,  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage, reportUntypedBaseClass]
):
    def __init__(
        self,
        token_cache: _TokenCache | dict[str, str],
        metrics: OperationalMetrics,
        logger: JsonLogger,
    ) -> None:
        if isinstance(token_cache, dict):
            token_cache = InMemoryAccessTokenCache(token_cache)
        self._token_cache: _TokenCache = token_cache
        self._metrics = metrics
        self._logger = logger

    def start_call(
        self,
        info: object,
        headers: Mapping[str, object],
    ) -> BearerAuthMiddleware:
        started = time.monotonic()
        method = str(cast(object, getattr(info, "method", "unknown")))
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

    def _reject(self, method: str, started: float, message: str) -> NoReturn:
        elapsed = time.monotonic() - started
        self._metrics.record_rpc(method, "UNAUTHENTICATED", elapsed)
        self._logger.event(
            "flight.rpc.completed",
            method=method,
            status="UNAUTHENTICATED",
            latencyMs=round(elapsed * 1000.0, 3),
        )
        # Never include the supplied authorization value in logs or errors.
        raise _unauthenticated_exception(f"UNAUTHENTICATED: {message}")


def authenticated_subject(context: _CallContext) -> str:
    middleware = context.get_middleware(MIDDLEWARE_KEY)
    subject = (
        None
        if middleware is None
        else cast(object, getattr(middleware, "subject", None))
    )
    if not isinstance(subject, str) or not subject:
        raise _unauthenticated_exception(
            "UNAUTHENTICATED: authentication middleware is unavailable"
        )
    return subject


def _single_authorization_header(
    headers: Mapping[str, object],
) -> str | None:
    values = headers.get("authorization")
    if not values:
        return None
    header_values: Sequence[object]
    if isinstance(values, (list, tuple)):
        header_values = cast(Sequence[object], values)
    else:
        header_values = (values,)
    if len(header_values) != 1:
        return None
    value = header_values[0]
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    return value if isinstance(value, str) else None


def _validate_credential(token: object, subject: object) -> None:
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


def validate_bearer_credentials(tokens: object) -> dict[str, str]:
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("at least one bearer token is required")
    credentials = cast(dict[object, object], tokens)
    for token, subject in credentials.items():
        _validate_credential(token, subject)
    return {
        cast(str, token): cast(str, subject)
        for token, subject in credentials.items()
    }


class InMemoryAccessTokenCache:
    """Small injection seam used by isolated middleware/server tests."""

    def __init__(self, tokens: dict[str, str]) -> None:
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


def _unauthenticated_exception(message: str) -> Exception:
    """Construct the runtime exception omitted by PyArrow's public stubs."""
    factory = cast(
        _FlightExceptionFactory,
        vars(flight)["FlightUnauthenticatedError"],
    )
    return factory(message)
