from __future__ import annotations

import hashlib
import json

_RECEIPT_FIELDS = (
    "payloadId",
    "ordinal",
    "schemaId",
    "dataContractSha256",
    "rows",
    "batches",
    "bytes",
    "sha256",
    "schemaFingerprint",
)


def canonical_receipts(inputs) -> list[dict]:
    receipts = []
    for item in sorted(inputs, key=lambda value: _field(value, "ordinal")):
        receipt = {
            "payloadId": _field(item, "payloadId", "payload_id"),
            "ordinal": _field(item, "ordinal"),
            "schemaId": _field(item, "schemaId", "schema_id"),
            "dataContractSha256": _field(
                item,
                "dataContractSha256",
                "data_contract_sha256",
            ),
            "rows": _field(item, "rows"),
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


def manifest_sha256(inputs) -> str:
    payload = json.dumps(
        canonical_receipts(inputs),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _field(document, *names):
    for name in names:
        if isinstance(document, dict) and name in document:
            return document[name]
        if hasattr(document, name):
            return getattr(document, name)
    raise ValueError(f"input receipt has no field {names[0]}")


__all__ = ["canonical_receipts", "manifest_sha256"]
