from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class MigrationStatusView(Protocol):
    @property
    def current(self) -> Sequence[str]: ...

    @property
    def heads(self) -> Sequence[str]: ...

    @property
    def pending(self) -> bool: ...


def print_status(status: MigrationStatusView) -> None:
    print(f"Current revision: {', '.join(status.current) or 'none'}")
    print(f"Head revision: {', '.join(status.heads) or 'none'}")
    print(f"Pending migrations: {'yes' if status.pending else 'no'}")


__all__ = ["MigrationStatusView", "print_status"]
