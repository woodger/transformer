from __future__ import annotations

from app.service.application.ports.access import AccessTokenRepository
from app.service.domain.access import AccessTokenRecord

_OWNER_SUBJECT = "inventory"


class AccessTokenAdministration:
    """Application use cases shared by administrative entrypoints."""

    def __init__(self, repository: AccessTokenRepository) -> None:
        self._repository = repository

    def issue(self) -> AccessTokenRecord:
        return self._repository.issue(_OWNER_SUBJECT)

    def list(self) -> list[AccessTokenRecord]:
        return self._repository.list()

    def revoke(self, token_id: str) -> AccessTokenRecord:
        return self._repository.revoke(token_id)


__all__ = ["AccessTokenAdministration"]
