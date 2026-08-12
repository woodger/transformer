from __future__ import annotations

from app.service.application.ports.access import AccessTokenRepository
from app.service.domain.access import AccessTokenRecord


class AccessTokenAdministration:
    """Application use cases shared by administrative entrypoints."""

    def __init__(self, repository: AccessTokenRepository) -> None:
        self._repository = repository

    def issue(self, subject: str) -> AccessTokenRecord:
        return self._repository.issue(subject)

    def list(self) -> list[AccessTokenRecord]:
        return self._repository.list()

    def revoke(self, token_id: str) -> AccessTokenRecord:
        return self._repository.revoke(token_id)


__all__ = ["AccessTokenAdministration"]
