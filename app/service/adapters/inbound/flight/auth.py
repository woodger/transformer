import re
import time
from collections.abc import Mapping, Sequence
from typing import NoReturn, Protocol, cast

import pyarrow.flight as flight

from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.application.ports.authentication import (
    AccessTokenAuthenticator,
    InvalidAccessTokenError,
)

MIDDLEWARE_KEY = "auth"
_BEARER_TOKEN = re.compile(r"^[\x21-\x7e]{1,4096}$")


class _CallContext(Protocol):
    def get_middleware(self, key: str) -> object | None: ...


class _FlightExceptionFactory(Protocol):
    def __call__(self, message: str) -> Exception: ...


class BearerAuthMiddleware(
    flight.ServerMiddleware,  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage, reportUntypedBaseClass]
):
    def __init__(
        self,
        owner_subject: str,
        method: str,
        metrics: OperationalMetrics,
        logger: JsonLogger,
        *,
        started: float,
    ) -> None:
        self.owner_subject = owner_subject
        self.method = method
        self._metrics = metrics
        self._logger = logger
        self._started = started

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
        authenticator: AccessTokenAuthenticator,
        metrics: OperationalMetrics,
        logger: JsonLogger,
    ) -> None:
        self._authenticator = authenticator
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
            self._reject(
                method,
                started,
                "UNAUTHENTICATED",
                "missing bearer authorization metadata",
                "FlightUnauthenticatedError",
            )
        scheme, separator, token = value.partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or not _BEARER_TOKEN.fullmatch(token)
        ):
            self._reject(
                method,
                started,
                "UNAUTHENTICATED",
                "invalid bearer authorization metadata",
                "FlightUnauthenticatedError",
            )

        try:
            principal = self._authenticator.authenticate(token)
        except InvalidAccessTokenError:
            self._reject(
                method,
                started,
                "UNAUTHENTICATED",
                "invalid bearer credential",
                "FlightUnauthenticatedError",
            )
        except Exception:
            # Fail closed and never expose an adapter error that could contain
            # the credential.
            self._reject(
                method,
                started,
                "UNAVAILABLE",
                "authentication service is unavailable",
                "FlightUnavailableError",
            )

        return BearerAuthMiddleware(
            owner_subject=principal.owner_subject,
            method=method,
            metrics=self._metrics,
            logger=self._logger,
            started=started,
        )

    def _reject(
        self,
        method: str,
        started: float,
        status: str,
        message: str,
        exception_name: str,
    ) -> NoReturn:
        elapsed = time.monotonic() - started
        self._metrics.record_rpc(method, status, elapsed)
        self._logger.event(
            "flight.rpc.completed",
            method=method,
            status=status,
            latencyMs=round(elapsed * 1000.0, 3),
        )
        raise _flight_exception(exception_name, f"{status}: {message}")


def authenticated_owner_subject(context: _CallContext) -> str:
    middleware = context.get_middleware(MIDDLEWARE_KEY)
    owner_subject = (
        None
        if middleware is None
        else cast(object, getattr(middleware, "owner_subject", None))
    )
    if not isinstance(owner_subject, str) or not owner_subject:
        raise _flight_exception(
            "FlightUnauthenticatedError",
            "UNAUTHENTICATED: authentication middleware is unavailable",
        )
    return owner_subject


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


def _flight_exception(name: str, message: str) -> Exception:
    """Construct a runtime exception omitted by PyArrow's public stubs."""
    factory = cast(_FlightExceptionFactory, vars(flight)[name])
    return factory(message)
