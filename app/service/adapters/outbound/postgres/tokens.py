from __future__ import annotations

import base64
import hashlib
import secrets
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.service.adapters.outbound.postgres.models import ApiAccessToken
from app.service.adapters.outbound.postgres.session import Database
from app.service.domain.access import AccessTokenRecord


class AccessTokenStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def issue(self, subject: str) -> AccessTokenRecord:
        _validate_subject(subject)
        token = _generate_token()
        record = ApiAccessToken(
            token_id=str(uuid.uuid4()),
            token_digest=_digest(token),
            subject=subject,
            created_at=datetime.now(UTC),
        )
        with self.database.transaction() as session:
            session.add(record)
            session.flush()
        return _record(record, token=token)

    def list(self) -> list[AccessTokenRecord]:
        with self.database.session() as session:
            records = session.scalars(
                select(ApiAccessToken).order_by(
                    ApiAccessToken.created_at,
                    ApiAccessToken.token_id,
                )
            )
            return [_record(record) for record in records]

    def revoke(self, token_id: str) -> AccessTokenRecord:
        token_id = _canonical_uuid(token_id)
        with self.database.transaction() as session:
            record = session.get(ApiAccessToken, token_id, with_for_update=True)
            if record is None:
                raise LookupError(f"API access token not found: {token_id}")
            if record.revoked_at is None:
                record.revoked_at = datetime.now(UTC)
                session.flush()
            return _record(record)

    def active_credentials(self) -> list[tuple[str, str, str]]:
        """Return credential digests for the in-process authentication cache."""
        with self.database.session() as session:
            records = session.scalars(
                select(ApiAccessToken)
                .where(ApiAccessToken.revoked_at.is_(None))
                .order_by(ApiAccessToken.token_id)
            )
            return [
                (record.token_digest, record.token_id, record.subject)
                for record in records
            ]


def _record(
    record: ApiAccessToken,
    *,
    token: str | None = None,
) -> AccessTokenRecord:
    return AccessTokenRecord(
        token_id=record.token_id,
        subject=record.subject,
        created_at=record.created_at,
        revoked_at=record.revoked_at,
        token=token,
    )


def _generate_token() -> str:
    encoded = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")
    return f"a.{encoded}"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _validate_subject(raw_subject: object) -> None:
    if (
        not isinstance(raw_subject, str)
        or not raw_subject
        or len(raw_subject) > 256
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in raw_subject
        )
    ):
        raise ValueError(
            "subject must be 1-256 characters without control characters"
        )


def _canonical_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError("token ID must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError("token ID must be a canonical UUID")
    return str(parsed)


__all__ = ["AccessTokenStore"]
