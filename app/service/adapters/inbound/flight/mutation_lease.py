from __future__ import annotations

import base64
import binascii
import json
from typing import cast


class MutationLeaseError(ValueError):
    """The opaque public lease cannot be resolved to a durable fence."""


def encode_mutation_lease(client_execution_id: str, fencing_token: int) -> str:
    payload = json.dumps(
        {"execution": client_execution_id, "fence": fencing_token},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def decode_mutation_lease(value: str) -> tuple[str, int]:
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        decoded_document: object = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, binascii.Error) as exc:
        raise MutationLeaseError("mutationLease is invalid") from exc
    if not isinstance(decoded_document, dict):
        raise MutationLeaseError("mutationLease is invalid")
    document = cast(dict[str, object], decoded_document)
    if set(document) != {"execution", "fence"}:
        raise MutationLeaseError("mutationLease is invalid")
    execution = document["execution"]
    fence = document["fence"]
    if not isinstance(execution, str):
        raise MutationLeaseError("mutationLease is invalid")
    if isinstance(fence, bool) or not isinstance(fence, int) or fence < 1:
        raise MutationLeaseError("mutationLease is invalid")
    return execution, fence


__all__ = [
    "MutationLeaseError",
    "decode_mutation_lease",
    "encode_mutation_lease",
]
