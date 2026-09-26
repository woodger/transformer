from __future__ import annotations

from dataclasses import dataclass

from app.service.domain.job import ErrorCode


@dataclass(frozen=True)
class AttemptExecutionError(Exception):
    """Безопасный сбой, сообщённый адаптером возможности выполнения."""

    code: ErrorCode
    message: str
    exit_code: int | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)


__all__ = ["AttemptExecutionError"]
