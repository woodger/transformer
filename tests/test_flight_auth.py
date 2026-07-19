from types import SimpleNamespace

import pyarrow.flight as flight
import pytest

from app.flight.auth import BearerAuthMiddlewareFactory
from app.flight.observability import JsonLogger, OperationalMetrics


def factory():
    return BearerAuthMiddlewareFactory(
        {"secret-token": "inventory"},
        OperationalMetrics(),
        JsonLogger("transformer.flight.auth-test"),
    )


def test_bearer_auth_accepts_configured_metadata():
    middleware = factory().start_call(
        SimpleNamespace(method="DoAction"),
        {"authorization": ["Bearer secret-token"]},
    )

    assert middleware.subject == "inventory"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"authorization": ["Basic secret-token"]},
        {"authorization": ["Bearer wrong"]},
        {"authorization": ["Bearer secret-token", "Bearer secret-token"]},
    ],
)
def test_bearer_auth_rejects_missing_or_invalid_credentials(headers):
    with pytest.raises(flight.FlightUnauthenticatedError):
        factory().start_call(SimpleNamespace(method="DoAction"), headers)
