from __future__ import annotations

import base64
import http.client
import json
import ssl
from collections.abc import Sequence
from typing import cast
from urllib.parse import urlsplit

from app.contracts.json_types import JsonObject
from app.service.adapters.outbound.opensearch.config import (
    OpenSearchMetricsConfig,
)
from app.service.application.ports.metrics import (
    BlockedMetricsDeliveryError,
    RetryableMetricsDeliveryError,
)

_RETRYABLE_STATUSES = frozenset({408, 429})


class OpenSearchMetricsClient:
    """Use only bounded Bulk create and conflict-verification reads."""

    def __init__(self, config: OpenSearchMetricsConfig) -> None:
        self.config = config
        parsed = urlsplit(config.endpoint)
        if parsed.hostname is None:
            raise ValueError("OpenSearch endpoint has no hostname")
        self._host = parsed.hostname
        self._use_tls = parsed.scheme == "https"
        self._port = parsed.port or (443 if self._use_tls else 80)
        self._authorization: str | None = None
        self._context: ssl.SSLContext | None = None
        if config.username is not None and config.password is not None:
            self._authorization = "Basic " + base64.b64encode(
                f"{config.username}:{config.password}".encode()
            ).decode("ascii")
        if self._use_tls:
            self._context = ssl.create_default_context(cafile=config.ca_file)

    def create_documents(
        self,
        index: str,
        documents: Sequence[JsonObject],
        *,
        id_field: str,
    ) -> None:
        if not documents:
            return
        if len(documents) > self.config.max_bulk_documents:
            raise ValueError("OpenSearch Bulk document limit exceeded")
        encoded = bytearray()
        identities: list[tuple[str, str]] = []
        for document in documents:
            document_id = _required_string(document, id_field)
            document_sha256 = _required_string(document, "documentSha256")
            identities.append((document_id, document_sha256))
            encoded.extend(_json_line({
                "create": {"_index": index, "_id": document_id}
            }))
            encoded.extend(_json_line(document))
        if len(encoded) > self.config.max_bulk_bytes:
            raise ValueError("OpenSearch Bulk byte limit exceeded")
        response = self._request("POST", "/_bulk", bytes(encoded), ndjson=True)
        items = response.get("items")
        if not isinstance(items, list) or len(items) != len(documents):
            raise RetryableMetricsDeliveryError(
                "OpenSearch Bulk response has no complete item list"
            )
        conflicts: list[tuple[str, str]] = []
        retryable: list[int] = []
        blocked: list[int] = []
        for item, identity in zip(items, identities, strict=True):
            if not isinstance(item, dict):
                raise RetryableMetricsDeliveryError(
                    "OpenSearch Bulk item is malformed"
                )
            create_value = item.get("create")
            if not isinstance(create_value, dict):
                raise RetryableMetricsDeliveryError(
                    "OpenSearch Bulk item is malformed"
                )
            create = cast(JsonObject, create_value)
            status = create.get("status")
            if status in (200, 201):
                continue
            if status == 409:
                conflicts.append(identity)
            elif isinstance(status, int) and _is_retryable_status(status):
                retryable.append(status)
            elif isinstance(status, int):
                blocked.append(status)
            else:
                retryable.append(0)
        if blocked:
            raise BlockedMetricsDeliveryError(
                f"OpenSearch rejected metrics documents with status {blocked[0]}"
            )
        if conflicts:
            self._verify_conflicts(index, conflicts)
        if retryable:
            raise RetryableMetricsDeliveryError(
                f"OpenSearch temporarily rejected metrics documents with status "
                f"{retryable[0]}"
            )

    def _verify_conflicts(
        self,
        index: str,
        conflicts: Sequence[tuple[str, str]],
    ) -> None:
        body = json.dumps(
            {
                "docs": [
                    {
                        "_id": document_id,
                        "_source": ["documentSha256"],
                    }
                    for document_id, _ in conflicts
                ],
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        response = self._request("POST", f"/{index}/_mget", body)
        documents = response.get("docs")
        if not isinstance(documents, list) or len(documents) != len(conflicts):
            raise RetryableMetricsDeliveryError(
                "OpenSearch conflict verification response is incomplete"
            )
        expected = dict(conflicts)
        for document in documents:
            if not isinstance(document, dict):
                raise RetryableMetricsDeliveryError(
                    "OpenSearch conflict verification item is malformed"
                )
            document_id = document.get("_id")
            source = document.get("_source")
            if (
                document.get("found") is not True
                or not isinstance(document_id, str)
                or document_id not in expected
                or not isinstance(source, dict)
                or source.get("documentSha256") != expected[document_id]
            ):
                raise BlockedMetricsDeliveryError(
                    "OpenSearch document identity conflict failed integrity check"
                )

    def _request(
        self,
        method: str,
        path: str,
        body: bytes,
        *,
        ndjson: bool = False,
    ) -> JsonObject:
        connection: http.client.HTTPConnection
        if self._use_tls:
            connection = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self.config.connect_timeout_seconds,
                context=self._context,
            )
        else:
            connection = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=self.config.connect_timeout_seconds,
            )
        headers = {
            "Content-Type": (
                "application/x-ndjson" if ndjson else "application/json"
            ),
        }
        if self._authorization is not None:
            headers["Authorization"] = self._authorization
        try:
            connection.connect()
            if connection.sock is not None:
                connection.sock.settimeout(self.config.request_timeout_seconds)
            connection.request(
                method,
                path,
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            payload = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise RetryableMetricsDeliveryError(
                "OpenSearch request failed"
            ) from exc
        finally:
            connection.close()
        if not 200 <= response.status < 300:
            error_type = (
                RetryableMetricsDeliveryError
                if _is_retryable_status(response.status)
                else BlockedMetricsDeliveryError
            )
            raise error_type(
                f"OpenSearch request returned status {response.status}"
            )
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetryableMetricsDeliveryError(
                "OpenSearch response is not valid JSON"
            ) from exc
        if not isinstance(document, dict):
            raise RetryableMetricsDeliveryError(
                "OpenSearch response is not an object"
            )
        return cast(JsonObject, document)


def _is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE_STATUSES or 500 <= status <= 599


def _json_line(document: JsonObject) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _required_string(document: JsonObject, field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"OpenSearch document field {field} must be a string")
    return value


__all__ = ["OpenSearchMetricsClient"]
