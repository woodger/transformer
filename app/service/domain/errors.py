from dataclasses import dataclass

from app.service.domain.job import ErrorCode


@dataclass(eq=False)
class ServiceError(Exception):
    code: ErrorCode
    message: str

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def safe_text(self) -> str:
        return f"{self.code.value}: {self.message}"


def invalid(message: str) -> ServiceError:
    return ServiceError(ErrorCode.INVALID_ARGUMENT, message)


def not_found(message: str) -> ServiceError:
    return ServiceError(ErrorCode.NOT_FOUND, message)


def failed_precondition(message: str) -> ServiceError:
    return ServiceError(ErrorCode.FAILED_PRECONDITION, message)


def conflict(message: str) -> ServiceError:
    return ServiceError(ErrorCode.ALREADY_EXISTS, message)


def resource_exhausted(message: str) -> ServiceError:
    return ServiceError(ErrorCode.RESOURCE_EXHAUSTED, message)

