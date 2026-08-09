from __future__ import annotations


def print_status(status) -> None:
    print(f"Current revision: {', '.join(status.current) or 'none'}")
    print(f"Head revision: {', '.join(status.heads) or 'none'}")
    print(f"Pending migrations: {'yes' if status.pending else 'no'}")


__all__ = ["print_status"]
