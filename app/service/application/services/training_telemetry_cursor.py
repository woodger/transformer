from __future__ import annotations

import base64
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast


class InvalidTrainingTelemetryCursor(ValueError):
    pass


class ExpiredTrainingTelemetryCursor(ValueError):
    pass


class InvalidatedTrainingTelemetryCursor(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingTelemetryCursor:
    operation: Literal["report", "gradientInteractions"]
    model_ref: str
    producing_run_id: str
    report_identity: str
    page_size: int
    after_epoch: int | None
    epoch: int | None
    after_pair: tuple[str, str] | None
    expires_at: datetime


class TrainingTelemetryCursorCodec:
    def __init__(self, key: bytes, *, boot_identity: str) -> None:
        if len(key) < 32:
            raise ValueError("telemetry cursor signing key must contain 32 bytes")
        if not _boot_identity(boot_identity):
            raise ValueError(
                "telemetry cursor boot identity must be 128-bit lowercase hex"
            )
        self._key = key
        self._boot_identity = boot_identity

    def encode(
        self,
        owner_subject: str,
        cursor: TrainingTelemetryCursor,
    ) -> str:
        gradient = cursor.operation == "gradientInteractions"
        prefix = "ttg1." if gradient else "tte1."
        payload = json.dumps(
            [
                1,
                cursor.operation,
                cursor.model_ref,
                cursor.producing_run_id,
                cursor.report_identity,
                cursor.page_size,
                cursor.after_epoch,
                cursor.epoch,
                None if cursor.after_pair is None else list(cursor.after_pair),
                _micros(cursor.expires_at),
                self._boot_identity,
            ],
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.digest(
            self._key,
            b"training-telemetry-v1\0"
            + owner_subject.encode("utf-8")
            + b"\0"
            + payload,
            "sha256",
        )
        return prefix + base64.urlsafe_b64encode(
            payload + signature
        ).rstrip(b"=").decode("ascii")

    def decode(
        self,
        owner_subject: str,
        token: str,
        *,
        operation: Literal["report", "gradientInteractions"],
        model_ref: str,
        producing_run_id: str,
        page_size: int,
        epoch: int | None,
        now: datetime,
    ) -> TrainingTelemetryCursor:
        prefix = "ttg1." if operation == "gradientInteractions" else "tte1."
        try:
            if not token.startswith(prefix):
                raise InvalidTrainingTelemetryCursor
            encoded = token[len(prefix) :]
            padding = "=" * (-len(encoded) % 4)
            envelope = base64.b64decode(
                encoded + padding,
                altchars=b"-_",
                validate=True,
            )
            if len(envelope) <= 32:
                raise InvalidTrainingTelemetryCursor
            payload, signature = envelope[:-32], envelope[-32:]
            expected = hmac.digest(
                self._key,
                b"training-telemetry-v1\0"
                + owner_subject.encode("utf-8")
                + b"\0"
                + payload,
                "sha256",
            )
            if not hmac.compare_digest(signature, expected):
                raise InvalidTrainingTelemetryCursor
            raw: object = json.loads(payload)
            if not isinstance(raw, list):
                raise InvalidTrainingTelemetryCursor
            values = cast(list[object], raw)
            if len(values) != 11:
                raise InvalidTrainingTelemetryCursor
            pair = _pair(values[8])
            if (
                values[0] != 1
                or values[1] != operation
                or values[2] != model_ref
                or values[3] != producing_run_id
                or not isinstance(values[4], str)
                or len(values[4]) != 64
                or not _integer(values[5])
                or values[5] != page_size
                or not _optional_integer(values[6])
                or not _optional_integer(values[7])
                or not _integer(values[9])
                or not _boot_identity(values[10])
            ):
                raise InvalidTrainingTelemetryCursor
            after_epoch = cast(int | None, values[6])
            encoded_epoch = cast(int | None, values[7])
            if operation == "report":
                if after_epoch is None or after_epoch < 1 or encoded_epoch is not None or pair is not None:
                    raise InvalidTrainingTelemetryCursor
            elif (
                epoch is None
                or encoded_epoch != epoch
                or after_epoch is not None
                or pair is None
            ):
                raise InvalidTrainingTelemetryCursor
            cursor = TrainingTelemetryCursor(
                operation=operation,
                model_ref=model_ref,
                producing_run_id=cast(str, values[3]),
                report_identity=values[4],
                page_size=cast(int, values[5]),
                after_epoch=after_epoch,
                epoch=encoded_epoch,
                after_pair=pair,
                expires_at=_from_micros(cast(int, values[9])),
            )
        except InvalidTrainingTelemetryCursor:
            raise
        except (
            UnicodeError,
            ValueError,
            TypeError,
            OverflowError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise InvalidTrainingTelemetryCursor from exc
        if now.astimezone(UTC) >= cursor.expires_at:
            raise ExpiredTrainingTelemetryCursor
        if values[10] != self._boot_identity:
            raise InvalidatedTrainingTelemetryCursor
        return cursor


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_integer(value: object) -> bool:
    return value is None or _integer(value)


def _boot_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _pair(value: object) -> tuple[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise InvalidTrainingTelemetryCursor
    items = cast(list[object], value)
    if len(items) != 2 or not all(
        isinstance(item, str) and item for item in items
    ):
        raise InvalidTrainingTelemetryCursor
    return cast(tuple[str, str], tuple(items))


def _micros(value: datetime) -> int:
    utc = value.astimezone(UTC)
    return int(utc.timestamp()) * 1_000_000 + utc.microsecond


def _from_micros(value: int) -> datetime:
    seconds, micros = divmod(value, 1_000_000)
    return datetime.fromtimestamp(seconds, UTC).replace(microsecond=micros)


__all__ = [
    "ExpiredTrainingTelemetryCursor",
    "InvalidTrainingTelemetryCursor",
    "InvalidatedTrainingTelemetryCursor",
    "TrainingTelemetryCursor",
    "TrainingTelemetryCursorCodec",
]
