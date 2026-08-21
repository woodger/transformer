import json
import time
from urllib.parse import parse_qs

import pytest

import app.service.adapters.outbound.hydra.client as client_module
from app.service.adapters.outbound.hydra.client import (
    HydraAccessTokenAuthenticator,
)
from app.service.adapters.outbound.hydra.config import (
    INTROSPECTION_TIMEOUT_SECONDS,
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


def _authenticator():
    return HydraAccessTokenAuthenticator(HydraIntrospectionConfig(
        "http://hp260g9.home:4445"
    ))


def test_configuration_accepts_only_the_hydra_admin_endpoint(tmp_path):
    config = load_hydra_introspection_config(
        environ={
            "HYDRA_ENDPOINT": "http://hp260g9.home:4445",
        },
        env_file=tmp_path / "absent.env",
    )

    assert config.endpoint == "http://hp260g9.home:4445"


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
    _install_http_response(monkeypatch, json.dumps(document).encode("utf-8"))

    with pytest.raises(AuthenticationUnavailableError):
        _authenticator().authenticate("opaque-token")


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
    _install_http_response(monkeypatch, payload, status=status)

    with pytest.raises(AuthenticationUnavailableError):
        _authenticator().authenticate("opaque-token")


def test_network_failure_is_unavailable_without_exposing_token(monkeypatch):
    _install_http_response(
        monkeypatch,
        b"",
        connect_error=TimeoutError("opaque-sensitive-token"),
    )

    with pytest.raises(AuthenticationUnavailableError) as error:
        _authenticator().authenticate("opaque-sensitive-token")

    assert "opaque-sensitive-token" not in str(error.value)
