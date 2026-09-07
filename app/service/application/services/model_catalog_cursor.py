from __future__ import annotations

import base64
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast


class InvalidCatalogCursor(ValueError):
    pass


class ExpiredCatalogCursor(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CatalogCursor:
    page_size: int
    high_water_ordinal: int
    after_created_at: datetime
    after_model_ref: str
    expires_at: datetime


class CatalogCursorCodec:
    _PREFIX = "mc1."

    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("catalog cursor signing key must contain 32 bytes")
        self._key = key

    def encode(self, owner_subject: str, cursor: CatalogCursor) -> str:
        payload = json.dumps(
            [
                1,
                cursor.page_size,
                cursor.high_water_ordinal,
                _micros(cursor.after_created_at),
                cursor.after_model_ref,
                _micros(cursor.expires_at),
            ],
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.digest(
            self._key,
            owner_subject.encode("utf-8") + b"\0" + payload,
            "sha256",
        )
        token = base64.urlsafe_b64encode(payload + signature).rstrip(b"=")
        return self._PREFIX + token.decode("ascii")

    def decode(
        self,
        owner_subject: str,
        token: str,
        *,
        page_size: int,
        now: datetime,
    ) -> CatalogCursor:
        try:
            if not token.startswith(self._PREFIX):
                raise InvalidCatalogCursor
            encoded = token[len(self._PREFIX):]
            padding = "=" * (-len(encoded) % 4)
            envelope = base64.b64decode(
                encoded + padding,
                altchars=b"-_",
                validate=True,
            )
            if len(envelope) <= 32:
                raise InvalidCatalogCursor
            payload, signature = envelope[:-32], envelope[-32:]
            expected = hmac.digest(
                self._key,
                owner_subject.encode("utf-8") + b"\0" + payload,
                "sha256",
            )
            if not hmac.compare_digest(signature, expected):
                raise InvalidCatalogCursor
            raw_value: object = json.loads(payload)
            if not isinstance(raw_value, list):
                raise InvalidCatalogCursor
            value = cast(list[object], raw_value)
            if (
                len(value) != 6
                or value[0] != 1
                or isinstance(value[1], bool)
                or not isinstance(value[1], int)
                or isinstance(value[2], bool)
                or not isinstance(value[2], int)
                or isinstance(value[3], bool)
                or not isinstance(value[3], int)
                or not isinstance(value[4], str)
                or isinstance(value[5], bool)
                or not isinstance(value[5], int)
                or value[1] != page_size
                or value[2] < 0
            ):
                raise InvalidCatalogCursor
            cursor = CatalogCursor(
                page_size=value[1],
                high_water_ordinal=value[2],
                after_created_at=_from_micros(value[3]),
                after_model_ref=value[4],
                expires_at=_from_micros(value[5]),
            )
        except InvalidCatalogCursor:
            raise
        except (
            UnicodeError,
            ValueError,
            TypeError,
            OverflowError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise InvalidCatalogCursor from exc
        if now.astimezone(UTC) >= cursor.expires_at:
            raise ExpiredCatalogCursor
        return cursor


def _micros(value: datetime) -> int:
    utc = value.astimezone(UTC)
    return int(utc.timestamp()) * 1_000_000 + utc.microsecond


def _from_micros(value: int) -> datetime:
    seconds, micros = divmod(value, 1_000_000)
    return datetime.fromtimestamp(seconds, UTC).replace(microsecond=micros)


__all__ = [
    "CatalogCursor",
    "CatalogCursorCodec",
    "ExpiredCatalogCursor",
    "InvalidCatalogCursor",
]
