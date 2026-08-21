import json
from types import SimpleNamespace

import pyarrow.flight as flight
import pytest

from app.service.adapters.inbound.flight.auth import BearerAuthMiddlewareFactory
from app.service.adapters.observability import OperationalMetrics
from app.service.application.ports.authentication import (
    AuthenticationUnavailableError,
    InsufficientAccessError,
    InvalidAccessTokenError,
)
from app.service.domain.authentication import AuthenticatedPrincipal
from tests.support.authentication import StaticAccessTokenAuthenticator


class RecordingLogger:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))


def factory(authenticator=None, logger=None):
    return BearerAuthMiddlewareFactory(
        authenticator
        or StaticAccessTokenAuthenticator({"opaque-token": "inventory"}),
        OperationalMetrics(),
        logger or RecordingLogger(),
    )


def test_bearer_auth_uses_hydra_principal_as_owner_identity():
    authenticator = StaticAccessTokenAuthenticator({
        "first-token": "inventory",
        "rotated-token": "inventory",
    })

    first = factory(authenticator).start_call(
        SimpleNamespace(method="DoAction"),
        {"authorization": ["Bearer first-token"]},
    )
    rotated = factory(authenticator).start_call(
        SimpleNamespace(method="DoGet"),
        {"authorization": ["Bearer rotated-token"]},
    )

    assert first.owner_subject == rotated.owner_subject == "inventory"
    assert authenticator.calls == ["first-token", "rotated-token"]


@pytest.mark.parametrize(
    ("headers", "introspection_expected"),
    [
        ({}, False),
        ({"authorization": ["Basic opaque-token"]}, False),
        ({"authorization": ["Bearer has whitespace"]}, False),
        (
            {"authorization": ["Bearer opaque-token", "Bearer opaque-token"]},
            False,
        ),
        ({"authorization": ["Bearer unknown-token"]}, True),
    ],
)
def test_missing_malformed_and_inactive_credentials_are_unauthenticated(
    headers,
    introspection_expected,
):
    authenticator = StaticAccessTokenAuthenticator({
        "opaque-token": "inventory",
    })

    with pytest.raises(
        flight.FlightUnauthenticatedError,
        match="UNAUTHENTICATED",
    ):
        factory(authenticator).start_call(
            SimpleNamespace(method="DoAction"),
            headers,
        )

    assert bool(authenticator.calls) is introspection_expected


@pytest.mark.parametrize(
    ("failure", "expected", "status"),
    [
        (
            InvalidAccessTokenError("sensitive token"),
            flight.FlightUnauthenticatedError,
            "UNAUTHENTICATED",
        ),
        (
            InsufficientAccessError("sensitive token"),
            flight.FlightUnauthorizedError,
            "PERMISSION_DENIED",
        ),
        (
            AuthenticationUnavailableError("sensitive token"),
            flight.FlightUnavailableError,
            "UNAVAILABLE",
        ),
        (
            RuntimeError("sensitive token"),
            flight.FlightUnavailableError,
            "UNAVAILABLE",
        ),
    ],
)
def test_authentication_outcomes_map_to_safe_flight_statuses(
    failure,
    expected,
    status,
):
    logger = RecordingLogger()

    class FailingAuthenticator:
        def authenticate(self, _access_token):
            raise failure

    with pytest.raises(expected, match=status) as error:
        factory(FailingAuthenticator(), logger).start_call(
            SimpleNamespace(method="DoPut"),
            {"authorization": ["Bearer opaque-sensitive-token"]},
        )

    serialized = json.dumps(logger.events)
    assert "opaque-sensitive-token" not in str(error.value)
    assert "opaque-sensitive-token" not in serialized
    assert "sensitive token" not in serialized


def test_one_authenticated_rpc_keeps_its_established_principal():
    class OneShotAuthenticator:
        def __init__(self):
            self.calls = 0

        def authenticate(self, _access_token):
            self.calls += 1
            if self.calls > 1:
                raise InvalidAccessTokenError("expired")
            return AuthenticatedPrincipal(owner_subject="inventory")

    authenticator = OneShotAuthenticator()
    middleware = factory(authenticator).start_call(
        SimpleNamespace(method="DoPut"),
        {"authorization": ["Bearer short-lived-token"]},
    )

    assert middleware.owner_subject == "inventory"
    middleware.call_completed(None)
    assert authenticator.calls == 1

    with pytest.raises(flight.FlightUnauthenticatedError):
        factory(authenticator).start_call(
            SimpleNamespace(method="DoAction"),
            {"authorization": ["Bearer short-lived-token"]},
        )
