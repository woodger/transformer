import hashlib
import re
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

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

    issued = AccessTokenStore(database).issue("inventory")

    assert uuid.UUID(issued.token_id).version == 4
    assert issued.subject == "inventory"
    assert issued.revoked_at is None
    assert issued.token is not None
    assert re.fullmatch(r"a\.[A-Za-z0-9_-]{86}", issued.token)
    stored = database.current.records[0]
    assert isinstance(stored, ApiAccessToken)
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


def test_list_omits_credentials_and_revoke_is_idempotent():
    database = _Database()
    store = AccessTokenStore(database)
    issued = store.issue("inventory")

    listed = store.list()
    first_revoke = store.revoke(issued.token_id)
    second_revoke = store.revoke(issued.token_id)

    assert len(listed) == 1
    assert listed[0].token is None
    assert first_revoke.revoked_at is not None
    assert second_revoke.revoked_at == first_revoke.revoked_at
    assert database.current.flushes == 2


def test_revoke_rejects_invalid_or_unknown_token_id():
    store = AccessTokenStore(_Database())

    with pytest.raises(ValueError, match="canonical UUID"):
        store.revoke("not-a-uuid")
    with pytest.raises(LookupError, match="not found"):
        store.revoke("12345678-1234-4234-8234-123456789abc")


def test_digest_cache_authenticates_subject_and_drops_revoked_entries():
    token = "a." + "A" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()

    class Store:
        credentials = [(digest, "token-id", "inventory")]

        def active_credentials(self):
            return self.credentials

    store = Store()
    cache = AccessTokenCache()

    assert cache.reload(store) == 1
    assert cache.authenticate(token).owner_subject == "inventory"
    assert token not in repr(vars(cache))

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
        "revoked_at",
    }
    assert ApiAccessToken.__table__.c.created_at.type.timezone is True
    assert datetime.now(UTC).tzinfo is UTC
