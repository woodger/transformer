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


class CreatedOAuthClientView(Protocol):
    @property
    def client_id(self) -> str: ...

    @property
    def client_secret(self) -> str: ...


def print_created(record: CreatedOAuthClientView) -> None:
    print(f"Client ID: {record.client_id}")
    print(f"Client Secret: {record.client_secret}")
    print("Audience: transformer")
    print("Scope: transformer:invoke")


def print_list(records: Sequence[OAuthClientView]) -> None:
    print("CLIENT ID\tCREATED AT\tSTATUS")
    for record in records:
        status = "ready" if record.ready else "misconfigured"
        print(f"{record.client_id}\t{record.created_at}\t{status}")


def print_deleted(client_id: str) -> None:
    print(f"Deleted OAuth client: {client_id}")


__all__ = [
    "CreatedOAuthClientView",
    "OAuthClientView",
    "print_created",
    "print_deleted",
    "print_list",
]
