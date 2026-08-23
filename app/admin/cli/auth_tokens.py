from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.service.domain.access import AccessTokenRecord


def print_issued(record: AccessTokenRecord) -> None:
    print(f"Token ID: {record.token_id}")
    print(f"Expires: {record.expires_at.isoformat()}")
    print(f"Token: {record.token}")


def print_list(
    records: Sequence[AccessTokenRecord],
    *,
    now: datetime | None = None,
) -> None:
    current_time = datetime.now(UTC) if now is None else now
    headers = ("ID", "Status", "Created", "Expires")
    rows = [
        (
            record.token_id,
            _status(record, current_time),
            record.created_at.isoformat(),
            record.expires_at.isoformat(),
        )
        for record in records
    ]
    widths = tuple(
        max(len(values[index]) for values in (headers, *rows))
        for index in range(len(headers))
    )
    for row in (headers, *rows):
        print(
            "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            ).rstrip()
        )


def _status(record: AccessTokenRecord, now: datetime) -> str:
    if record.expires_at <= now:
        return "Expired"
    return "Active"


def print_revoked(record: AccessTokenRecord) -> None:
    print(f"Revoked token: {record.token_id}")


__all__ = ["print_issued", "print_list", "print_revoked"]
