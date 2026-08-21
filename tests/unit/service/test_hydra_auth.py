import json
import queue
import threading
import time
from urllib.parse import parse_qs

import pytest

import app.service.adapters.outbound.hydra.client as client_module
from app.service.adapters.outbound.hydra.client import (
    HydraAccessTokenAuthenticator,
)
from app.service.adapters.outbound.hydra.config import (
    INTROSPECTION_TIMEOUT_SECONDS,
    HydraAuthorizationCacheConfig,
    HydraIntrospectionConfig,
    load_hydra_introspection_config,
)
from app.service.application.ports.authentication import (
    AuthenticationUnavailableError,
    InsufficientAccessError,
    InvalidAccessTokenError,
)


def _active_document(**overrides):
    document = {
        "active": True,
        "client_id": "inventory",
        "aud": ["transformer"],
        "scope": "openid transformer:invoke extra",
        "token_type": "Bearer",
        "exp": int(time.time()) + 3600,
    }
    document.update(overrides)
    return document


def _install_http_response(monkeypatch, payload, *, status=200, connect_error=None):
    requests = []
    connections = []

    class Socket:
        def __init__(self):
            self.timeout = None

        def settimeout(self, value):
            self.timeout = value

    class Response:
        def __init__(self):
            self.status = status

        def read(self, limit):
            assert limit == 64 * 1024 + 1
            return payload

    class Connection:
        def __init__(self, host, port, timeout):
            self.sock = Socket()
            connections.append((host, port, timeout, self))

        def connect(self):
            if connect_error is not None:
                raise connect_error

        def request(self, method, path, body, headers):
            requests.append((method, path, body, headers))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(client_module.http.client, "HTTPConnection", Connection)
    return connections, requests


def _authenticator(
    *,
    positive_ttl_seconds=15.0,
    negative_ttl_seconds=2.0,
    max_entries=1024,
):
    return HydraAccessTokenAuthenticator(
        HydraIntrospectionConfig(
            "http://hp260g9.home:4445",
            authorization_cache=HydraAuthorizationCacheConfig(
                positive_ttl_seconds=positive_ttl_seconds,
                negative_ttl_seconds=negative_ttl_seconds,
                max_entries=max_entries,
            ),
        )
    )


def test_configuration_accepts_only_the_hydra_admin_endpoint(tmp_path):
    config = load_hydra_introspection_config(
        environ={
            "HYDRA_ENDPOINT": "http://hp260g9.home:4445",
        },
        env_file=tmp_path / "absent.env",
    )

    assert config.endpoint == "http://hp260g9.home:4445"
    assert config.authorization_cache == HydraAuthorizationCacheConfig(
        positive_ttl_seconds=15.0,
        negative_ttl_seconds=2.0,
        max_entries=1024,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"positive_ttl_seconds": 0.0},
        {"positive_ttl_seconds": float("inf")},
        {"negative_ttl_seconds": -1.0},
        {"max_entries": 0},
        {"max_entries": True},
    ],
)
def test_authorization_cache_configuration_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        HydraAuthorizationCacheConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {},
        {
            "ORY_HYDRA_INTROSPECTION_ENDPOINT": (
                "http://hp260g9.home:4445/admin/oauth2/introspect"
            ),
        },
        {
            "HYDRA_ENDPOINT": (
                "http://hp260g9.home:4445/admin/oauth2/introspect"
            ),
        },
        {
            "HYDRA_ENDPOINT": "http://hp260g9.home:4445",
            "ORY_HYDRA_SCOPE": "configurable:scope",
        },
    ],
)
def test_configuration_rejects_missing_or_variable_contract_settings(
    tmp_path,
    values,
):
    with pytest.raises(ValueError):
        load_hydra_introspection_config(
            environ=values,
            env_file=tmp_path / "absent.env",
        )


def test_introspection_is_form_encoded_and_returns_exact_client_id(monkeypatch):
    payload = json.dumps(_active_document()).encode("utf-8")
    connections, requests = _install_http_response(monkeypatch, payload)

    principal = _authenticator().authenticate("opaque +/= token")

    assert principal.owner_subject == "inventory"
    host, port, timeout, connection = connections[0]
    assert (host, port, timeout) == (
        "hp260g9.home",
        4445,
        INTROSPECTION_TIMEOUT_SECONDS,
    )
    assert connection.sock.timeout == INTROSPECTION_TIMEOUT_SECONDS
    method, path, body, headers = requests[0]
    assert (method, path) == ("POST", "/admin/oauth2/introspect")
    assert parse_qs(body.decode("ascii")) == {
        "token": ["opaque +/= token"],
    }
    assert "scope" not in parse_qs(body.decode("ascii"))
    assert headers == {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def test_successful_authorization_is_cached_without_retaining_raw_token(
    monkeypatch,
):
    payload = json.dumps(_active_document()).encode("utf-8")
    _, requests = _install_http_response(monkeypatch, payload)
    authenticator = _authenticator()
    access_token = "opaque-sensitive-token"

    first = authenticator.authenticate(access_token)
    second = authenticator.authenticate(access_token)

    assert first == second
    assert len(requests) == 1
    assert access_token not in repr(vars(authenticator))


def test_cached_authorization_survives_hydra_outage_until_ttl(monkeypatch):
    payload = json.dumps(_active_document()).encode("utf-8")
    _install_http_response(monkeypatch, payload)
    authenticator = _authenticator()
    expected = authenticator.authenticate("opaque-token")
    failed_connections, _ = _install_http_response(
        monkeypatch,
        b"",
        connect_error=TimeoutError("Hydra unavailable"),
    )

    actual = authenticator.authenticate("opaque-token")

    assert actual == expected
    assert failed_connections == []


def test_positive_cache_expires_after_configured_ttl(monkeypatch):
    clock = {"monotonic": 10.0, "wall": 1_000.0}
    monkeypatch.setattr(
        client_module.time,
        "monotonic",
        lambda: clock["monotonic"],
    )
    monkeypatch.setattr(
        client_module.time,
        "time",
        lambda: clock["wall"],
    )
    payload = json.dumps(_active_document(exp=2_000)).encode("utf-8")
    _, requests = _install_http_response(monkeypatch, payload)
    authenticator = _authenticator()

    authenticator.authenticate("opaque-token")
    clock["monotonic"] += 14.0
    clock["wall"] += 14.0
    authenticator.authenticate("opaque-token")
    assert len(requests) == 1

    clock["monotonic"] += 1.0
    clock["wall"] += 1.0
    authenticator.authenticate("opaque-token")
    assert len(requests) == 2


def test_positive_cache_never_outlives_token_expiry(monkeypatch):
    clock = {"monotonic": 10.0, "wall": 1_000.0}
    monkeypatch.setattr(
        client_module.time,
        "monotonic",
        lambda: clock["monotonic"],
    )
    monkeypatch.setattr(
        client_module.time,
        "time",
        lambda: clock["wall"],
    )
    payload = json.dumps(_active_document(exp=1_005)).encode("utf-8")
    _, requests = _install_http_response(monkeypatch, payload)
    authenticator = _authenticator()

    authenticator.authenticate("opaque-token")
    clock["monotonic"] += 4.0
    clock["wall"] += 4.0
    authenticator.authenticate("opaque-token")
    assert len(requests) == 1

    clock["monotonic"] += 1.0
    clock["wall"] += 1.0
    with pytest.raises(InvalidAccessTokenError):
        authenticator.authenticate("opaque-token")
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("document", "expected_error"),
    [
        ({"active": False}, InvalidAccessTokenError),
        (
            _active_document(scope="transformer:read"),
            InsufficientAccessError,
        ),
    ],
)
def test_definitive_rejection_is_cached_for_two_seconds(
    monkeypatch,
    document,
    expected_error,
):
    clock = {"monotonic": 10.0, "wall": 1_000.0}
    monkeypatch.setattr(
        client_module.time,
        "monotonic",
        lambda: clock["monotonic"],
    )
    monkeypatch.setattr(
        client_module.time,
        "time",
        lambda: clock["wall"],
    )
    payload = json.dumps(document).encode("utf-8")
    _, requests = _install_http_response(monkeypatch, payload)
    authenticator = _authenticator()

    for _ in range(2):
        with pytest.raises(expected_error):
            authenticator.authenticate("opaque-token")
    assert len(requests) == 1

    clock["monotonic"] += 2.0
    clock["wall"] += 2.0
    with pytest.raises(expected_error):
        authenticator.authenticate("opaque-token")
    assert len(requests) == 2


def test_authorization_cache_uses_lru_eviction(monkeypatch):
    payload = json.dumps(_active_document()).encode("utf-8")
    _, requests = _install_http_response(monkeypatch, payload)
    authenticator = _authenticator(max_entries=2)

    authenticator.authenticate("token-a")
    authenticator.authenticate("token-b")
    authenticator.authenticate("token-a")
    authenticator.authenticate("token-c")
    authenticator.authenticate("token-b")

    assert len(requests) == 4


def test_parallel_requests_share_one_introspection(monkeypatch):
    authenticator = _authenticator()
    waiter_started = threading.Event()
    start = threading.Barrier(3)
    calls = []
    results = queue.Queue()
    original_future_result = client_module.Future.result

    def tracked_future_result(future, timeout=None):
        waiter_started.set()
        return original_future_result(future, timeout)

    def introspect(access_token):
        calls.append(access_token)
        assert waiter_started.wait(timeout=1.0)
        return _active_document()

    def authenticate():
        start.wait()
        try:
            results.put(authenticator.authenticate("shared-token"))
        except BaseException as exc:
            results.put(exc)

    monkeypatch.setattr(client_module.Future, "result", tracked_future_result)
    monkeypatch.setattr(authenticator, "_introspect", introspect)
    threads = [threading.Thread(target=authenticate) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    outcomes = [results.get_nowait(), results.get_nowait()]
    assert all(
        outcome.owner_subject == "inventory"
        for outcome in outcomes
    )
    assert calls == ["shared-token"]


@pytest.mark.parametrize(
    "document",
    [
        {"active": False},
        _active_document(token_type="refresh_token"),
        _active_document(exp=1),
    ],
)
def test_inactive_expired_and_non_access_credentials_are_rejected(
    monkeypatch,
    document,
):
    _install_http_response(monkeypatch, json.dumps(document).encode("utf-8"))

    with pytest.raises(InvalidAccessTokenError):
        _authenticator().authenticate("opaque-token")


@pytest.mark.parametrize(
    "document",
    [
        _active_document(aud=["inventory"]),
        _active_document(scope="transformer:read"),
        _active_document(aud=None),
        _active_document(scope=None),
    ],
)
def test_active_token_requires_exact_audience_and_scope(monkeypatch, document):
    _install_http_response(monkeypatch, json.dumps(document).encode("utf-8"))

    with pytest.raises(InsufficientAccessError):
        _authenticator().authenticate("opaque-token")


@pytest.mark.parametrize(
    "document",
    [
        {"active": "true"},
        _active_document(client_id=""),
        _active_document(token_type=None),
        _active_document(aud=123),
        _active_document(scope=["transformer:invoke"]),
        _active_document(exp=1.5),
    ],
)
def test_malformed_active_response_is_unavailable(monkeypatch, document):
    connections, _ = _install_http_response(
        monkeypatch,
        json.dumps(document).encode("utf-8"),
    )
    authenticator = _authenticator()

    for _ in range(2):
        with pytest.raises(AuthenticationUnavailableError):
            authenticator.authenticate("opaque-token")

    assert len(connections) == 2


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (b"{}", 503),
        (b"not-json", 200),
        (b"[]", 200),
        (b"x" * (64 * 1024 + 1), 200),
    ],
)
def test_non_success_and_invalid_responses_are_unavailable(
    monkeypatch,
    payload,
    status,
):
    connections, _ = _install_http_response(
        monkeypatch,
        payload,
        status=status,
    )
    authenticator = _authenticator()

    for _ in range(2):
        with pytest.raises(AuthenticationUnavailableError):
            authenticator.authenticate("opaque-token")

    assert len(connections) == 2


def test_network_failure_is_unavailable_without_exposing_token(monkeypatch):
    connections, _ = _install_http_response(
        monkeypatch,
        b"",
        connect_error=TimeoutError("opaque-sensitive-token"),
    )
    authenticator = _authenticator()

    for _ in range(2):
        with pytest.raises(AuthenticationUnavailableError) as error:
            authenticator.authenticate("opaque-sensitive-token")

        assert "opaque-sensitive-token" not in str(error.value)

    assert len(connections) == 2
