from __future__ import annotations

import http.client
import json
import ssl
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import parse_qs, quote, urlencode, urlsplit

from app.contracts.json_types import JsonObject
from app.service.adapters.outbound.hydra.config import (
    INTROSPECTION_TIMEOUT_SECONDS,
    REQUIRED_AUDIENCE,
    REQUIRED_SCOPE,
)
from app.service.domain.authentication import is_valid_owner_subject

_CLIENTS_PATH = "/admin/clients"
_TOKENS_PATH = "/admin/oauth2/tokens"
_MANAGED_OWNER = "transformer-auth-clients"
_PAGE_SIZE = 100
_MAX_RESPONSE_BYTES = 1024 * 1024


class HydraAdministrationError(RuntimeError):
    """A safe operator-facing Hydra administration failure."""


@dataclass(frozen=True, slots=True)
class OAuthClientRecord:
    client_id: str
    created_at: str
    ready: bool


@dataclass(frozen=True, slots=True)
class CreatedOAuthClient:
    client_id: str
    client_secret: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Response:
    status: int
    payload: bytes
    link: str | None


class HydraOAuthClientAdministration:
    """Manage Transformer-scoped OAuth clients through Hydra Admin API."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = INTROSPECTION_TIMEOUT_SECONDS,
    ) -> None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "Hydra administration endpoint must be an HTTP(S) base URL"
            )
        self._host = parsed.hostname
        self._use_tls = parsed.scheme == "https"
        self._port = parsed.port or (443 if self._use_tls else 80)
        self._timeout_seconds = timeout_seconds
        self._tls_context = (
            ssl.create_default_context() if self._use_tls else None
        )

    def create(self, client_id: str) -> CreatedOAuthClient:
        _require_client_id(client_id)
        body = json.dumps(
            {
                "access_token_strategy": "opaque",
                "audience": [REQUIRED_AUDIENCE],
                "client_id": client_id,
                "client_name": client_id,
                "grant_types": ["client_credentials"],
                "owner": _MANAGED_OWNER,
                "response_types": ["token"],
                "scope": REQUIRED_SCOPE,
                "token_endpoint_auth_method": "client_secret_basic",
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        response = self._request("POST", _CLIENTS_PATH, body=body)
        if response.status == 409:
            raise HydraAdministrationError(
                f"Hydra OAuth client already exists: {client_id}"
            )
        _require_status(response, {201}, "create OAuth client")
        document = _decode_object(response.payload)
        actual_client_id = document.get("client_id")
        client_secret = document.get("client_secret")
        if actual_client_id != client_id or not isinstance(client_secret, str):
            raise HydraAdministrationError(
                "Hydra create-client response is incomplete"
            )
        if not client_secret:
            raise HydraAdministrationError(
                "Hydra create-client response has no client secret"
            )
        return CreatedOAuthClient(
            client_id=client_id,
            client_secret=client_secret,
        )

    def list(self) -> list[OAuthClientRecord]:
        records: list[OAuthClientRecord] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            query = {
                "owner": _MANAGED_OWNER,
                "page_size": str(_PAGE_SIZE),
            }
            if page_token is not None:
                query["page_token"] = page_token
            response = self._request(
                "GET",
                f"{_CLIENTS_PATH}?{urlencode(query)}",
            )
            _require_status(response, {200}, "list OAuth clients")
            documents = _decode_array(response.payload)
            records.extend(_client_record(document) for document in documents)
            next_token = _next_page_token(response.link)
            if next_token is None:
                break
            if next_token in seen_tokens:
                raise HydraAdministrationError(
                    "Hydra client pagination contains a cycle"
                )
            seen_tokens.add(next_token)
            page_token = next_token
        return sorted(records, key=lambda record: record.client_id)

    def delete(self, client_id: str) -> str:
        _require_client_id(client_id)
        client_path = f"{_CLIENTS_PATH}/{quote(client_id, safe='')}"
        response = self._request("GET", client_path)
        if response.status == 404:
            self._delete_access_tokens(client_id)
            return client_id
        _require_status(response, {200}, "read OAuth client")
        document = _decode_object(response.payload)
        if document.get("owner") != _MANAGED_OWNER:
            raise HydraAdministrationError(
                f"Hydra OAuth client is not managed by Transformer: {client_id}"
            )

        self._delete_access_tokens(client_id)
        response = self._request("DELETE", client_path)
        _require_status(response, {204}, "delete OAuth client")
        self._delete_access_tokens(client_id)
        return client_id

    def _delete_access_tokens(self, client_id: str) -> None:
        query = urlencode({"client_id": client_id})
        response = self._request("DELETE", f"{_TOKENS_PATH}?{query}")
        _require_status(response, {204}, "delete OAuth access tokens")

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
    ) -> _Response:
        connection: http.client.HTTPConnection
        if self._use_tls:
            connection = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self._timeout_seconds,
                context=self._tls_context,
            )
        else:
            connection = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=self._timeout_seconds,
            )
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.connect()
            if connection.sock is not None:
                connection.sock.settimeout(self._timeout_seconds)
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
            status = response.status
            link = response.getheader("Link")
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise HydraAdministrationError(
                "Hydra Admin API is unavailable"
            ) from exc
        finally:
            connection.close()
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise HydraAdministrationError(
                "Hydra Admin API response is too large"
            )
        return _Response(status=status, payload=payload, link=link)


def _require_client_id(client_id: str) -> None:
    if not is_valid_owner_subject(client_id):
        raise ValueError("OAuth client ID is invalid")


def _require_status(
    response: _Response,
    expected: set[int],
    operation: str,
) -> None:
    if response.status not in expected:
        raise HydraAdministrationError(
            f"Hydra failed to {operation}: HTTP {response.status}"
        )


def _decode_object(payload: bytes) -> JsonObject:
    document = _decode_json(payload)
    if not isinstance(document, dict):
        raise HydraAdministrationError("Hydra response is not an object")
    return cast(JsonObject, document)


def _decode_array(payload: bytes) -> list[JsonObject]:
    documents = _decode_json(payload)
    if not isinstance(documents, list):
        raise HydraAdministrationError("Hydra response is not an object list")
    items = cast(list[object], documents)
    if not all(isinstance(document, dict) for document in items):
        raise HydraAdministrationError("Hydra response is not an object list")
    return cast(list[JsonObject], items)


def _decode_json(payload: bytes) -> object:
    try:
        return cast(object, json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HydraAdministrationError(
            "Hydra response is not valid JSON"
        ) from exc


def _client_record(document: JsonObject) -> OAuthClientRecord:
    client_id = document.get("client_id")
    created_at = document.get("created_at")
    if not is_valid_owner_subject(client_id) or not isinstance(created_at, str):
        raise HydraAdministrationError(
            "Hydra OAuth client document is incomplete"
        )
    if document.get("owner") != _MANAGED_OWNER:
        raise HydraAdministrationError(
            "Hydra returned an OAuth client outside the requested owner"
        )
    return OAuthClientRecord(
        client_id=client_id,
        created_at=created_at,
        ready=_has_transformer_access(document),
    )


def _has_transformer_access(document: JsonObject) -> bool:
    audience = document.get("audience")
    grant_types = document.get("grant_types")
    scope = document.get("scope")
    return (
        isinstance(audience, list)
        and REQUIRED_AUDIENCE in audience
        and isinstance(grant_types, list)
        and "client_credentials" in grant_types
        and isinstance(scope, str)
        and REQUIRED_SCOPE in scope.split()
        and document.get("access_token_strategy") == "opaque"
        and document.get("token_endpoint_auth_method")
        == "client_secret_basic"
    )


def _next_page_token(link: str | None) -> str | None:
    if not link:
        return None
    for item in link.split(","):
        if 'rel="next"' not in item:
            continue
        start = item.find("<")
        end = item.find(">", start + 1)
        if start < 0 or end < 0:
            raise HydraAdministrationError(
                "Hydra client pagination link is malformed"
            )
        parsed = urlsplit(item[start + 1 : end])
        if parsed.path != _CLIENTS_PATH:
            raise HydraAdministrationError(
                "Hydra client pagination link has an unexpected path"
            )
        values = parse_qs(parsed.query).get("page_token")
        if values is None or len(values) != 1 or not values[0]:
            raise HydraAdministrationError(
                "Hydra client pagination link has no page token"
            )
        return values[0]
    return None


__all__ = [
    "CreatedOAuthClient",
    "HydraAdministrationError",
    "HydraOAuthClientAdministration",
    "OAuthClientRecord",
]
