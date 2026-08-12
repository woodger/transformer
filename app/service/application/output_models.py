from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputTicketGrant:
    token: bytes
    expires_at: float


__all__ = ["OutputTicketGrant"]
