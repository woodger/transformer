from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

from app.contracts.json_types import JsonObject, JsonValue
from app.service.adapters.inbound.flight.errors import invalid
from app.service.domain.input_manifest import canonical_receipts, manifest_sha256

MAX_ACTION_DOCUMENT_BYTES = 64 * 1024


@runtime_checkable
class _ArrowBuffer(Protocol):
    def to_pybytes(self) -> bytes: ...


def parse_action_body(body: object) -> JsonObject:
    if isinstance(body, _ArrowBuffer):
        body = body.to_pybytes()
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise invalid("action body must be UTF-8 JSON bytes")
    if isinstance(body, bytes):
        body_bytes = body
    elif isinstance(body, bytearray):
        body_bytes = bytes(body)
    else:
        body_bytes = body.tobytes()
    if len(body_bytes) > MAX_ACTION_DOCUMENT_BYTES:
        raise invalid(
            f"action body exceeds {MAX_ACTION_DOCUMENT_BYTES} bytes"
        )
    try:
        document = json.loads(
            body_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise invalid(
            "action body must be a valid UTF-8 JSON document"
        ) from exc
    if not isinstance(document, dict):
        raise invalid("action body must be a JSON object")
    return cast(JsonObject, document)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def response_document(request_id: str, **fields: object) -> JsonObject:
    return {
        "requestId": request_id,
        **{key: _json_value(value, key) for key, value in fields.items()},
    }


def encode_document(document: JsonObject | list[JsonValue]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_request_hash(document: JsonObject) -> str:
    canonical = {
        key: value
        for key, value in document.items()
        if key not in ("requestId", "idempotencyKey")
    }
    return hashlib.sha256(encode_document(canonical)).hexdigest()


def canonical_manifest_receipts(
    inputs: Iterable[object],
) -> list[dict[str, object]]:
    return canonical_receipts(inputs)


def canonical_manifest_hash(inputs: Iterable[object]) -> str:
    return manifest_sha256(inputs)


def _json_value(value: object, location: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError(f"{location} contains a non-string object key")
        return {
            cast(str, key): _json_value(item, f"{location}.{key}")
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(cast(Sequence[object], value))
        ]
    raise TypeError(f"{location} is not JSON-serializable")


__all__ = [
    "MAX_ACTION_DOCUMENT_BYTES",
    "canonical_manifest_hash",
    "canonical_manifest_receipts",
    "canonical_request_hash",
    "encode_document",
    "parse_action_body",
    "response_document",
]
