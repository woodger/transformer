from __future__ import annotations

import base64
import hashlib
import secrets
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update

from app.service.adapters.outbound.postgres.models import ApiAccessToken
from app.service.adapters.outbound.postgres.session import Database
from app.service.domain.access import (
    AccessTokenRecord,
    AuthIdentity,
    access_token_expiration,
)


class AccessTokenStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def issue(
        self,
        subject: str,
        *,
        now: datetime | None = None,
    ) -> AccessTokenRecord:
        _validate_subject(subject)
        token = _generate_token()
        created_at = datetime.now(UTC) if now is None else now
        record = ApiAccessToken(
            token_id=str(uuid.uuid4()),
            token_digest=_digest(token),
            subject=subject,
            created_at=created_at,
            expires_at=access_token_expiration(created_at),
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
            deleted = _record(record)
            session.delete(record)
            session.flush()
            return deleted

    def use_active_credential(
        self,
        token_digest: str,
        *,
        now: datetime | None = None,
    ) -> AuthIdentity | None:
        current_time = datetime.now(UTC) if now is None else now
        with self.database.transaction() as session:
            record = session.scalar(
                update(ApiAccessToken)
                .where(
                    ApiAccessToken.token_digest == token_digest,
                    ApiAccessToken.expires_at > current_time,
                )
                .values(last_used_at=current_time)
                .returning(ApiAccessToken)
            )
            if record is None:
                return None
            return AuthIdentity(
                token_id=record.token_id,
                subject=record.subject,
                expires_at=record.expires_at,
            )


def _record(
    record: ApiAccessToken,
    *,
    token: str | None = None,
) -> AccessTokenRecord:
    return AccessTokenRecord(
        token_id=record.token_id,
        subject=record.subject,
        created_at=record.created_at,
        expires_at=record.expires_at,
        last_used_at=record.last_used_at,
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
