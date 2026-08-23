import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

import app.admin.adapters.outbound.hydra as hydra_module
import app.admin.bootstrap.auth_tokens as auth_tokens_command
from app.admin.adapters.outbound.hydra import (
    HydraAdministrationError,
    HydraOAuthClientAdministration,
    IssuedOAuthClient,
    OAuthClientRecord,
)


class _Socket:
    def __init__(self):
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout


class _Response:
    def __init__(self, status, document=None, *, link=None):
        self.status = status
        self.payload = (
            b"" if document is None else json.dumps(document).encode("utf-8")
        )
        self.link = link

    def read(self, _amount):
        return self.payload

    def getheader(self, name):
        return self.link if name == "Link" else None


def _install_http_responses(monkeypatch, responses):
    connections = []
    requests = []
    pending = list(responses)

    class Connection:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.sock = _Socket()
            connections.append(self)

        def connect(self):
            return None

        def request(self, method, path, body=None, headers=None):
            requests.append((method, path, body, headers))

        def getresponse(self):
            return pending.pop(0)

        def close(self):
            return None

    monkeypatch.setattr(hydra_module.http.client, "HTTPConnection", Connection)
    return connections, requests


def _managed_client(client_id, **overrides):
    document = {
        "access_token_strategy": "opaque",
        "audience": ["transformer"],
        "client_id": client_id,
        "created_at": "2026-08-23T10:00:00Z",
        "grant_types": ["client_credentials"],
        "owner": "transformer-auth-tokens",
        "scope": "transformer:invoke",
        "token_endpoint_auth_method": "client_secret_basic",
    }
    document.update(overrides)
    return document


def test_issue_creates_exact_transformer_client_and_returns_secret_once(
    monkeypatch,
):
    connections, requests = _install_http_responses(
        monkeypatch,
        [_Response(201, {"client_id": "consumer", "client_secret": "secret"})],
    )

    issued = HydraOAuthClientAdministration(
        "http://hp260g9.home:4445"
    ).issue("consumer")

    assert issued.client_id == "consumer"
    assert issued.client_secret == "secret"
    assert "secret" not in repr(issued)
    assert len(connections) == 1
    assert (connections[0].host, connections[0].port) == (
        "hp260g9.home",
        4445,
    )
    method, path, body, headers = requests[0]
    assert (method, path) == ("POST", "/admin/clients")
    assert json.loads(body) == {
        "access_token_strategy": "opaque",
        "audience": ["transformer"],
        "client_id": "consumer",
        "client_name": "consumer",
        "grant_types": ["client_credentials"],
        "owner": "transformer-auth-tokens",
        "response_types": ["token"],
        "scope": "transformer:invoke",
        "token_endpoint_auth_method": "client_secret_basic",
    }
    assert headers == {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def test_list_follows_hydra_pagination_and_reports_configuration_drift(
    monkeypatch,
):
    _, requests = _install_http_responses(
        monkeypatch,
        [
            _Response(
                200,
                [_managed_client("zeta")],
                link=(
                    "</admin/clients?page_size=100&page_token=next%2Fpage>; "
                    'rel="next"'
                ),
            ),
            _Response(
                200,
                [_managed_client("alpha", audience=["another-service"])],
            ),
        ],
    )

    records = HydraOAuthClientAdministration(
        "http://hp260g9.home:4445"
    ).list()

    assert records == [
        OAuthClientRecord(
            client_id="alpha",
            created_at="2026-08-23T10:00:00Z",
            ready=False,
        ),
        OAuthClientRecord(
            client_id="zeta",
            created_at="2026-08-23T10:00:00Z",
            ready=True,
        ),
    ]
    assert len(requests) == 2
    first_query = parse_qs(urlsplit(requests[0][1]).query)
    second_query = parse_qs(urlsplit(requests[1][1]).query)
    assert first_query == {
        "owner": ["transformer-auth-tokens"],
        "page_size": ["100"],
    }
    assert second_query == {
        "owner": ["transformer-auth-tokens"],
        "page_size": ["100"],
        "page_token": ["next/page"],
    }


def test_revoke_deletes_client_tokens_before_and_after_client_deletion(
    monkeypatch,
):
    _, requests = _install_http_responses(
        monkeypatch,
        [
            _Response(200, _managed_client("consumer/name")),
            _Response(204),
            _Response(204),
            _Response(204),
        ],
    )

    revoked = HydraOAuthClientAdministration(
        "http://hp260g9.home:4445"
    ).revoke("consumer/name")

    assert revoked == "consumer/name"
    assert [(method, path) for method, path, _, _ in requests] == [
        ("GET", "/admin/clients/consumer%2Fname"),
        ("DELETE", "/admin/oauth2/tokens?client_id=consumer%2Fname"),
        ("DELETE", "/admin/clients/consumer%2Fname"),
        ("DELETE", "/admin/oauth2/tokens?client_id=consumer%2Fname"),
    ]


def test_revoke_rejects_client_not_managed_by_transformer(monkeypatch):
    _, requests = _install_http_responses(
        monkeypatch,
        [_Response(200, _managed_client("foreign", owner="another-owner"))],
    )

    with pytest.raises(HydraAdministrationError, match="not managed"):
        HydraOAuthClientAdministration(
            "http://hp260g9.home:4445"
        ).revoke("foreign")

    assert len(requests) == 1


def test_hydra_error_does_not_expose_response_body(monkeypatch):
    _install_http_responses(
        monkeypatch,
        [_Response(500, {"client_secret": "must-not-leak"})],
    )

    with pytest.raises(HydraAdministrationError) as exc:
        HydraOAuthClientAdministration(
            "http://hp260g9.home:4445"
        ).issue("consumer")

    assert "must-not-leak" not in str(exc.value)


def test_auth_token_command_prints_issued_credentials(monkeypatch, capsys):
    class Administration:
        def __init__(self, endpoint):
            assert endpoint == "http://hydra-admin:4445"

        def issue(self, client_id):
            assert client_id == "consumer"
            return IssuedOAuthClient(client_id, "one-time-secret")

    monkeypatch.setattr(
        auth_tokens_command,
        "load_hydra_introspection_config",
        lambda: SimpleNamespace(endpoint="http://hydra-admin:4445"),
    )
    monkeypatch.setattr(
        auth_tokens_command,
        "HydraOAuthClientAdministration",
        Administration,
    )

    auth_tokens_command.run(
        SimpleNamespace(tokens_action="issue", client_id="consumer")
    )

    assert capsys.readouterr().out == (
        "Token ID: consumer\n"
        "Client ID: consumer\n"
        "Client Secret: one-time-secret\n"
        "Audience: transformer\n"
        "Scope: transformer:invoke\n"
    )


def test_auth_token_command_generates_client_id_when_omitted(
    monkeypatch,
    capsys,
):
    class Administration:
        def __init__(self, endpoint):
            assert endpoint == "http://hydra-admin:4445"

        def issue(self, client_id):
            assert client_id == "trf-0123456789abcdefabcd"
            return IssuedOAuthClient(client_id, "one-time-secret")

    monkeypatch.setattr(
        auth_tokens_command,
        "load_hydra_introspection_config",
        lambda: SimpleNamespace(endpoint="http://hydra-admin:4445"),
    )
    monkeypatch.setattr(
        auth_tokens_command,
        "HydraOAuthClientAdministration",
        Administration,
    )
    monkeypatch.setattr(
        auth_tokens_command.secrets,
        "token_hex",
        lambda size: "0123456789abcdefabcd" if size == 10 else "unexpected",
    )

    auth_tokens_command.run(
        SimpleNamespace(tokens_action="issue", client_id=None)
    )

    assert capsys.readouterr().out == (
        "Token ID: trf-0123456789abcdefabcd\n"
        "Client ID: trf-0123456789abcdefabcd\n"
        "Client Secret: one-time-secret\n"
        "Audience: transformer\n"
        "Scope: transformer:invoke\n"
    )
