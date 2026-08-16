from types import SimpleNamespace

import pyarrow.flight as flight
import pytest

from app.service.adapters.inbound.flight.auth import BearerAuthMiddlewareFactory
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.postgres.token_cache import AccessTokenCache


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


def test_new_rpc_uses_preloaded_digest_cache_without_reloading_store():
    class CredentialStore:
        def __init__(self):
            self.loads = 0

        def active_credentials(self):
            self.loads += 1
            return [("a.cached-token", "token-id", "inventory")]

    store = CredentialStore()
    cache = AccessTokenCache()
    assert cache.reload(store) == 1
    middleware_factory = BearerAuthMiddlewareFactory(
        cache,
        OperationalMetrics(),
        JsonLogger("transformer.flight.auth-cache-test"),
    )

    for method in ("DoAction", "DoGet"):
        middleware = middleware_factory.start_call(
            SimpleNamespace(method=method),
            {"authorization": ["Bearer a.cached-token"]},
        )
        assert middleware.subject == "inventory"

    assert store.loads == 1
