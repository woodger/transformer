from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from app.service.adapters.inbound.flight.constants import (
    CONTRACT_PATH_VERSION,
    MAX_PAYLOADS_PER_JOB,
)
from app.service.adapters.inbound.flight.errors import invalid


@dataclass(frozen=True, slots=True)
class JobDataDescriptor:
    job_id: str
    ordinal: int


def parse_input_descriptor(descriptor: object) -> JobDataDescriptor:
    parts = _descriptor_parts(descriptor)
    if (
        len(parts) != 6
        or parts[:3] != ("transformer", CONTRACT_PATH_VERSION, "jobs")
        or parts[4] != "inputs"
    ):
        raise invalid("invalid input Flight descriptor")
    return JobDataDescriptor(
        job_id=_uuid_value(parts[3], "descriptor jobId"),
        ordinal=_ordinal_text(parts[5], "descriptor ordinal"),
    )


def parse_output_descriptor(descriptor: object) -> JobDataDescriptor:
    parts = _descriptor_parts(descriptor)
    if (
        len(parts) != 6
        or parts[:3] != ("transformer", CONTRACT_PATH_VERSION, "jobs")
        or parts[4] != "outputs"
    ):
        raise invalid("invalid output Flight descriptor")
    return JobDataDescriptor(
        job_id=_uuid_value(parts[3], "descriptor jobId"),
        ordinal=_ordinal_text(parts[5], "descriptor ordinal"),
    )


def _descriptor_parts(descriptor: object) -> tuple[str, ...]:
    path = cast(object, getattr(descriptor, "path", None))
    if path is None:
        raise invalid("Flight descriptor must be a path descriptor")
    if not isinstance(path, Sequence):
        raise invalid("Flight descriptor path must be a sequence")
    parts: list[str] = []
    for value in cast(Sequence[object], path):
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise invalid("Flight descriptor path must be UTF-8") from exc
        if not isinstance(value, str):
            raise invalid("Flight descriptor path must contain strings")
        parts.append(value)
    return tuple(parts)


def _uuid_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise invalid(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise invalid(f"{label} must be a UUID string") from exc
    if str(parsed) != value.lower():
        raise invalid(f"{label} must be a canonical UUID string")
    return str(parsed)


def _ordinal_text(value: str, label: str) -> int:
    if (
        not value
        or len(value) > 20
        or not value.isascii()
        or not value.isdigit()
    ):
        raise invalid(f"{label} must be a non-negative integer")
    parsed = int(value)
    if str(parsed) != value:
        raise invalid(f"{label} must use canonical decimal notation")
    if parsed >= MAX_PAYLOADS_PER_JOB:
        raise invalid(f"{label} must be less than {MAX_PAYLOADS_PER_JOB}")
    return parsed


__all__ = [
    "JobDataDescriptor",
    "parse_input_descriptor",
    "parse_output_descriptor",
]
