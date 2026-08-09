from __future__ import annotations

from collections.abc import Sequence

from app.service.domain.access import AccessTokenRecord


def print_issued(record: AccessTokenRecord) -> None:
    print(f"Token ID: {record.token_id}")
    print(f"Subject: {record.subject}")
    print(f"Token: {record.token}")


def print_list(records: Sequence[AccessTokenRecord]) -> None:
    print("TOKEN ID\tSUBJECT\tCREATED AT\tSTATUS")
    for record in records:
        status = "revoked" if record.revoked_at is not None else "active"
        print(
            f"{record.token_id}\t{record.subject}\t"
            f"{record.created_at.isoformat()}\t{status}"
        )


def print_revoked(record: AccessTokenRecord) -> None:
    print(f"Revoked token: {record.token_id}")


__all__ = ["print_issued", "print_list", "print_revoked"]
