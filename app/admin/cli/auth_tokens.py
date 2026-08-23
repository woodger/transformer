from __future__ import annotations

from collections.abc import Sequence

from app.service.domain.access import AccessTokenRecord


def print_issued(record: AccessTokenRecord) -> None:
    print(f"Token ID: {record.token_id}")
    print(f"Token: {record.token}")


def print_list(records: Sequence[AccessTokenRecord]) -> None:
    headers = ("TOKEN ID", "CREATED AT", "STATUS")
    rows = [
        (
            record.token_id,
            record.created_at.isoformat(),
            "revoked" if record.revoked_at is not None else "active",
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


def print_revoked(record: AccessTokenRecord) -> None:
    print(f"Revoked token: {record.token_id}")


__all__ = ["print_issued", "print_list", "print_revoked"]
