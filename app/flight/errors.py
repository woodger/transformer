from dataclasses import dataclass

import pyarrow as pa
import pyarrow.flight as flight

from app.flight.constants import ErrorCode


@dataclass(eq=False)
class ServiceError(Exception):
    code: ErrorCode
    message: str

    def __post_init__(self):
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


def to_flight_exception(error: ServiceError) -> Exception:
    """Map service failures to the closest status exposed by PyArrow 24.

    PyArrow has dedicated Flight exceptions for authentication, authorization,
    cancellation, availability and internal failures. Arrow status exceptions
    provide the remaining public mappings (Invalid, KeyError and Capacity).
    The stable application code is kept in every error message.
    """
    text = error.safe_text()
    if error.code == ErrorCode.UNAUTHENTICATED:
        return flight.FlightUnauthenticatedError(text)
    if error.code == ErrorCode.PERMISSION_DENIED:
        return flight.FlightUnauthorizedError(text)
    if error.code == ErrorCode.NOT_FOUND:
        return pa.ArrowKeyError(text)
    if error.code in (ErrorCode.RESOURCE_EXHAUSTED, ErrorCode.DISK_FULL):
        return pa.ArrowCapacityError(text)
    if error.code == ErrorCode.CANCELLED:
        return flight.FlightCancelledError(text)
    if error.code in (ErrorCode.UNAVAILABLE, ErrorCode.DEVICE_UNAVAILABLE):
        return flight.FlightUnavailableError(text)
    if error.code in (
        ErrorCode.INTERNAL,
        ErrorCode.EXECUTION_INTERRUPTED,
        ErrorCode.SUBPROCESS_FAILED,
        ErrorCode.SUBPROCESS_HUNG,
        ErrorCode.MALFORMED_OUTPUT,
        ErrorCode.CUDA_OUT_OF_MEMORY,
    ):
        return flight.FlightInternalError(text)
    return pa.ArrowInvalid(text)
