from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class OAuthClientView(Protocol):
    @property
    def client_id(self) -> str: ...

    @property
    def created_at(self) -> str: ...

    @property
    def ready(self) -> bool: ...


class IssuedOAuthClientView(Protocol):
    @property
    def client_id(self) -> str: ...

    @property
    def client_secret(self) -> str: ...


def print_issued(record: IssuedOAuthClientView) -> None:
    print(f"Token ID: {record.client_id}")
    print(f"Client ID: {record.client_id}")
    print(f"Client Secret: {record.client_secret}")
    print("Audience: transformer")
    print("Scope: transformer:invoke")


def print_list(records: Sequence[OAuthClientView]) -> None:
    print("TOKEN ID\tCREATED AT\tSTATUS")
    for record in records:
        status = "ready" if record.ready else "misconfigured"
        print(f"{record.client_id}\t{record.created_at}\t{status}")


def print_revoked(client_id: str) -> None:
    print(f"Revoked OAuth client: {client_id}")


__all__ = [
    "IssuedOAuthClientView",
    "OAuthClientView",
    "print_issued",
    "print_list",
    "print_revoked",
]
