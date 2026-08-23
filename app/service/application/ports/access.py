from __future__ import annotations

from typing import Protocol

from app.service.domain.access import AccessTokenRecord


class AccessTokenRepository(Protocol):
    def issue(self, subject: str) -> AccessTokenRecord: ...

    def list(self) -> list[AccessTokenRecord]: ...

    def revoke(self, token_id: str) -> AccessTokenRecord: ...


__all__ = ["AccessTokenRepository"]
