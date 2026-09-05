from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import cast

_RECEIPT_FIELDS = (
    "payloadId",
    "ordinal",
    "schemaId",
    "dataContractSha256",
    "chunks",
    "logicalRows",
    "nativeRows",
    "firstRangeOrdinal",
    "firstExampleOffset",
    "lastRangeOrdinal",
    "nextExampleOffset",
    "batches",
    "bytes",
    "sha256",
    "schemaFingerprint",
)


def canonical_receipts(inputs: Iterable[object]) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for item in sorted(inputs, key=_ordinal):
        receipt: dict[str, object] = {
            "payloadId": _field(item, "payloadId", "payload_id"),
            "ordinal": _field(item, "ordinal"),
            "schemaId": _field(item, "schemaId", "schema_id"),
            "dataContractSha256": _field(
                item,
                "dataContractSha256",
                "data_contract_sha256",
            ),
            "chunks": _field(item, "chunks"),
            "logicalRows": _field(item, "logicalRows", "rows"),
            "nativeRows": list(cast(tuple[int, ...], _field(
                item,
                "nativeRows",
                "native_rows",
            ))),
            "firstRangeOrdinal": _field(
                item,
                "firstRangeOrdinal",
                "first_range_ordinal",
            ),
            "firstExampleOffset": _field(
                item,
                "firstExampleOffset",
                "first_example_offset",
            ),
            "lastRangeOrdinal": _field(
                item,
                "lastRangeOrdinal",
                "last_range_ordinal",
            ),
            "nextExampleOffset": _field(
                item,
                "nextExampleOffset",
                "next_example_offset",
            ),
            "batches": _field(item, "batches"),
            "bytes": _field(item, "bytes", "byte_count"),
            "sha256": _field(item, "sha256"),
            "schemaFingerprint": _field(
                item,
                "schemaFingerprint",
                "schema_fingerprint",
            ),
        }
        if tuple(receipt) != _RECEIPT_FIELDS:
            raise AssertionError("canonical input receipt fields changed")
        receipts.append(receipt)
    return receipts


def manifest_sha256(inputs: Iterable[object]) -> str:
    payload = json.dumps(
        canonical_receipts(inputs),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordinal(document: object) -> int:
    value = _field(document, "ordinal")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("input receipt ordinal must be an integer")
    return value


def _field(document: object, *names: str) -> object:
    if isinstance(document, Mapping):
        mapping = cast(Mapping[object, object], document)
        for name in names:
            if name in mapping:
                return mapping[name]

    record = cast(object, document)
    for name in names:
        if hasattr(record, name):
            # Canonicalization accepts wire-shaped mappings and immutable
            # receipt records without making either representation canonical.
            return cast(object, getattr(record, name))
    raise ValueError(f"input receipt has no field {names[0]}")


__all__ = ["canonical_receipts", "manifest_sha256"]
