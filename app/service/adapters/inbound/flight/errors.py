from typing import Protocol, cast

import pyarrow as pa
import pyarrow.flight as flight

from app.service.domain.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    invalid,
    not_found,
    resource_exhausted,
)
from app.service.domain.job import ErrorCode

__all__ = [
    "ServiceError",
    "conflict",
    "failed_precondition",
    "invalid",
    "not_found",
    "resource_exhausted",
    "to_flight_exception",
]


class _ExceptionFactory(Protocol):
    def __call__(self, message: str) -> Exception: ...


def to_flight_exception(error: ServiceError) -> Exception:
    """Map service failures to the closest status exposed by PyArrow 24.

    PyArrow has dedicated Flight exceptions for authentication, authorization,
    cancellation, availability and internal failures. Arrow status exceptions
    provide the remaining public mappings (Invalid, KeyError and Capacity).
    The stable application code is kept in every error message.
    """
    text = error.safe_text()
    if error.code == ErrorCode.UNAUTHENTICATED:
        return _flight_exception("FlightUnauthenticatedError", text)
    if error.code == ErrorCode.PERMISSION_DENIED:
        return _flight_exception("FlightUnauthorizedError", text)
    if error.code == ErrorCode.NOT_FOUND:
        return pa.ArrowKeyError(text)
    if error.code in (ErrorCode.RESOURCE_EXHAUSTED, ErrorCode.DISK_FULL):
        return pa.ArrowCapacityError(text)
    if error.code == ErrorCode.CANCELLED:
        return _flight_exception("FlightCancelledError", text)
    if error.code in (ErrorCode.UNAVAILABLE, ErrorCode.DEVICE_UNAVAILABLE):
        return _flight_exception("FlightUnavailableError", text)
    if error.code in (
        ErrorCode.INTERNAL,
        ErrorCode.DEVICE_LOST,
        ErrorCode.EXECUTION_INTERRUPTED,
        ErrorCode.SUBPROCESS_FAILED,
        ErrorCode.SUBPROCESS_HUNG,
        ErrorCode.MALFORMED_OUTPUT,
        ErrorCode.CUDA_OUT_OF_MEMORY,
        ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
        ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
        ErrorCode.RECOVERY_INPUT_UNAVAILABLE,
    ):
        return _flight_exception("FlightInternalError", text)
    return pa.ArrowInvalid(text)


def _flight_exception(name: str, message: str) -> Exception:
    """Construct a runtime Flight exception missing from PyArrow's stubs."""
    factory = cast(_ExceptionFactory, vars(flight)[name])
    return factory(message)
