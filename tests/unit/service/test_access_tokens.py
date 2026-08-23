import hashlib
import re
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from app.service.adapters.outbound.postgres.models import ApiAccessToken
from app.service.adapters.outbound.postgres.token_cache import AccessTokenCache
from app.service.adapters.outbound.postgres.tokens import AccessTokenStore
from app.service.application.commands.access_tokens import AccessTokenAdministration
from app.service.application.ports.authentication import InvalidAccessTokenError


class _Session:
    def __init__(self):
        self.records = []
        self.flushes = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def add(self, record):
        self.records.append(record)

    def flush(self):
        self.flushes += 1

    def delete(self, record):
        self.records.remove(record)

    def scalars(self, _statement):
        return list(self.records)

    def get(self, _model, token_id, *, with_for_update=False):
        assert with_for_update is True
        return next(
            (record for record in self.records if record.token_id == token_id),
            None,
        )


class _Database:
    def __init__(self):
        self.current = _Session()

    @contextmanager
    def transaction(self):
        yield self.current

    def session(self):
        return self.current


def test_issue_generates_token_without_storing_credential():
    database = _Database()
    created_at = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)

    issued = AccessTokenStore(database).issue("inventory", now=created_at)

    assert uuid.UUID(issued.token_id).version == 4
    assert issued.subject == "inventory"
    assert issued.created_at == created_at
    assert issued.expires_at == datetime(2026, 4, 30, 12, 30, tzinfo=UTC)
    assert issued.token is not None
    assert re.fullmatch(r"a\.[A-Za-z0-9_-]{86}", issued.token)
    stored = database.current.records[0]
    assert isinstance(stored, ApiAccessToken)
    assert stored.expires_at == issued.expires_at
    assert stored.token_digest == hashlib.sha256(
        issued.token.encode("ascii")
    ).hexdigest()
    assert not hasattr(stored, "token")
    assert issued.token not in repr(issued)


def test_administration_issues_tokens_for_inventory_owner():
    issued = AccessTokenAdministration(AccessTokenStore(_Database())).issue()

    assert issued.subject == "inventory"


@pytest.mark.parametrize("subject", ["", "x" * 257, "bad\nsubject"])
def test_issue_rejects_invalid_subject(subject):
    with pytest.raises(ValueError, match="subject must be"):
        AccessTokenStore(_Database()).issue(subject)


def test_list_omits_credentials_and_revoke_physically_deletes_token():
    database = _Database()
    store = AccessTokenStore(database)
    issued = store.issue("inventory")

    listed = store.list()
    deleted = store.revoke(issued.token_id)

    assert len(listed) == 1
    assert listed[0].token is None
    assert deleted.token_id == issued.token_id
    assert store.list() == []
    with pytest.raises(LookupError, match="not found"):
        store.revoke(issued.token_id)
    assert database.current.flushes == 2


def test_revoke_rejects_invalid_or_unknown_token_id():
    store = AccessTokenStore(_Database())

    with pytest.raises(ValueError, match="canonical UUID"):
        store.revoke("not-a-uuid")
    with pytest.raises(LookupError, match="not found"):
        store.revoke("12345678-1234-4234-8234-123456789abc")


def test_digest_cache_authenticates_subject_and_drops_deleted_entries():
    token = "a." + "A" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    expires_at = datetime(2026, 11, 23, 12, 30, tzinfo=UTC)

    class Store:
        credentials = [(digest, "token-id", "inventory", expires_at)]

        def active_credentials(self):
            return self.credentials

    store = Store()
    cache = AccessTokenCache()

    assert cache.reload(store) == 1
    assert cache.authenticate(
        token,
        now=expires_at - timedelta(microseconds=1),
    ).owner_subject == "inventory"
    assert token not in repr(vars(cache))

    with pytest.raises(InvalidAccessTokenError):
        cache.authenticate(token, now=expires_at)

    store.credentials = []
    assert cache.reload(store) == 0
    with pytest.raises(InvalidAccessTokenError):
        cache.authenticate(token)


def test_access_token_model_contains_digest_only():
    columns = set(ApiAccessToken.__table__.columns.keys())

    assert columns == {
        "token_id",
        "token_digest",
        "subject",
        "created_at",
        "expires_at",
    }
    assert ApiAccessToken.__table__.c.created_at.type.timezone is True
    assert ApiAccessToken.__table__.c.expires_at.type.timezone is True
    assert {
        constraint.name for constraint in ApiAccessToken.__table__.constraints
    } >= {
        "api_access_tokens_digest_format_ck",
        "api_access_tokens_expiry_order_ck",
    }
    active_index = next(
        index
        for index in ApiAccessToken.__table__.indexes
        if index.name == "api_access_tokens_active_idx"
    )
    assert [column.name for column in active_index.columns] == ["expires_at"]
    assert active_index.dialect_options["postgresql"]["where"] is None
    assert datetime.now(UTC).tzinfo is UTC
