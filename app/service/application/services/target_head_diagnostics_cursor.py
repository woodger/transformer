from __future__ import annotations

import base64
import hmac
import json
from datetime import UTC, datetime
from typing import cast


class TargetHeadDiagnosticsCursorError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_PREFIX = "thd1."
_REVISION = 1


def encode_target_head_diagnostics_cursor(
    key: bytes,
    owner_subject: str,
    *,
    model_ref: str,
    producing_run_id: str,
    report_identity: str,
    page_size: int,
    after_epoch: int,
    expires_at: datetime,
    boot_identity: str,
) -> str:
    payload = json.dumps(
        [
            _REVISION,
            model_ref,
            producing_run_id,
            report_identity,
            page_size,
            after_epoch,
            _micros(expires_at),
            boot_identity,
        ],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.digest(
        key,
        b"target-head-diagnostics-v1\0"
        + owner_subject.encode("utf-8")
        + b"\0"
        + payload,
        "sha256",
    )
    return _PREFIX + base64.urlsafe_b64encode(
        payload + signature
    ).rstrip(b"=").decode("ascii")


def decode_target_head_diagnostics_cursor(
    key: bytes,
    owner_subject: str,
    token: str,
    *,
    model_ref: str,
    producing_run_id: str,
    page_size: int,
    now: datetime,
    boot_identity: str,
) -> dict[str, object]:
    try:
        if not token.startswith(_PREFIX):
            raise TargetHeadDiagnosticsCursorError("invalid")
        encoded = token[len(_PREFIX) :]
        envelope = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
        if len(envelope) <= 32:
            raise TargetHeadDiagnosticsCursorError("invalid")
        payload, signature = envelope[:-32], envelope[-32:]
        expected = hmac.digest(
            key,
            b"target-head-diagnostics-v1\0"
            + owner_subject.encode("utf-8")
            + b"\0"
            + payload,
            "sha256",
        )
        if not hmac.compare_digest(signature, expected):
            raise TargetHeadDiagnosticsCursorError("invalid")
        raw: object = json.loads(payload)
        if not isinstance(raw, list):
            raise TargetHeadDiagnosticsCursorError("invalid")
        values = cast(list[object], raw)
        if len(values) != 8:
            raise TargetHeadDiagnosticsCursorError("invalid")
        (
            revision,
            encoded_model_ref,
            encoded_run_id,
            report_identity,
            encoded_page_size,
            after_epoch,
            expiration_micros,
            cursor_boot_identity,
        ) = values
        if (
            revision != _REVISION
            or encoded_model_ref != model_ref
            or encoded_run_id != producing_run_id
            or not _sha256(report_identity)
            or not _positive_integer(encoded_page_size)
            or encoded_page_size != page_size
            or not _positive_integer(after_epoch)
            or not _nonnegative_integer(expiration_micros)
            or not _boot_identity(cursor_boot_identity)
        ):
            raise TargetHeadDiagnosticsCursorError("invalid")
        decoded = {
            "reportIdentity": report_identity,
            "afterEpoch": after_epoch,
            "expiresAt": _from_micros(cast(int, expiration_micros)),
            "bootIdentity": cursor_boot_identity,
        }
    except TargetHeadDiagnosticsCursorError:
        raise
    except (
        UnicodeError,
        ValueError,
        TypeError,
        OverflowError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise TargetHeadDiagnosticsCursorError("invalid") from exc
    if now.astimezone(UTC) >= cast(datetime, decoded["expiresAt"]):
        raise TargetHeadDiagnosticsCursorError("expired")
    if decoded["bootIdentity"] != boot_identity:
        raise TargetHeadDiagnosticsCursorError("invalidated")
    return decoded


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _boot_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _micros(value: datetime) -> int:
    utc = value.astimezone(UTC)
    return int(utc.timestamp()) * 1_000_000 + utc.microsecond


def _from_micros(value: int) -> datetime:
    seconds, micros = divmod(value, 1_000_000)
    return datetime.fromtimestamp(seconds, UTC).replace(microsecond=micros)


__all__ = [
    "TargetHeadDiagnosticsCursorError",
    "decode_target_head_diagnostics_cursor",
    "encode_target_head_diagnostics_cursor",
]
