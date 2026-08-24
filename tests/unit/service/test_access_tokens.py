import hashlib
import math
import queue
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.dialects import postgresql

import app.service.adapters.outbound.postgres.token_cache as token_cache_module
from app.config import (
    ACCESS_TOKEN_CACHE_MAX_ENTRIES,
    ACCESS_TOKEN_CACHE_TTL_SECONDS,
)
from app.service.adapters.outbound.postgres.models import ApiAccessToken
from app.service.adapters.outbound.postgres.token_cache import AccessTokenCache
from app.service.adapters.outbound.postgres.tokens import AccessTokenStore
from app.service.application.commands.access_tokens import AccessTokenAdministration
from app.service.application.ports.authentication import InvalidAccessTokenError
from app.service.domain.access import AuthIdentity


class _Session:
    def __init__(self):
        self.records = []
        self.flushes = 0
        self.scalar_statements = []

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

    def scalar(self, statement):
        self.scalar_statements.append(statement)
        return self.records[0] if self.records else None

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


class _CredentialStore:
    def __init__(self, credentials=None):
        self.credentials = {} if credentials is None else credentials
        self.lookups = []

    def use_active_credential(self, token_digest, *, now):
        self.lookups.append((token_digest, now))
        identity = self.credentials.get(token_digest)
        if identity is None or identity.expires_at <= now:
            return None
        return identity


def _identity(expires_at, *, token_id="token-id", subject="inventory"):
    return AuthIdentity(
        token_id=token_id,
        subject=subject,
        expires_at=expires_at,
    )


def test_issue_generates_token_without_storing_credential():
    database = _Database()
    created_at = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)

    issued = AccessTokenStore(database).issue("inventory", now=created_at)

    assert uuid.UUID(issued.token_id).version == 4
    assert issued.subject == "inventory"
    assert issued.created_at == created_at
    assert issued.last_used_at is None
    assert issued.expires_at == datetime(2026, 7, 30, 12, 30, tzinfo=UTC)
    assert issued.expires_at - issued.created_at == timedelta(days=180)
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


def test_cache_aside_loads_on_first_use_and_reuses_positive_entry():
    token = "a." + "A" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = _CredentialStore({
        digest: _identity(now + timedelta(hours=1)),
    })
    cache = AccessTokenCache(store)

    assert store.lookups == []
    assert ACCESS_TOKEN_CACHE_TTL_SECONDS == 60.0
    assert ACCESS_TOKEN_CACHE_MAX_ENTRIES == 1024
    assert cache.authenticate(
        token,
        now=now,
        monotonic_now=100.0,
    ).owner_subject == "inventory"
    assert cache.authenticate(
        token,
        now=now + timedelta(seconds=59, microseconds=999_000),
        monotonic_now=159.999,
    ).owner_subject == "inventory"
    assert [lookup[0] for lookup in store.lookups] == [digest]
    assert token not in repr(vars(cache))


def test_cache_aside_uses_postgresql_token_store_identity():
    database = _Database()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = AccessTokenStore(database)
    issued = store.issue("inventory", now=now)
    cache = AccessTokenCache(store)

    assert issued.token is not None
    assert cache.authenticate(
        issued.token,
        now=now,
        monotonic_now=1.0,
    ).owner_subject == "inventory"
    statement = database.current.scalar_statements[-1]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = " ".join(str(compiled).split())
    assert sql.startswith(
        "UPDATE transformer.api_access_tokens SET last_used_at="
    )
    assert "api_access_tokens.token_digest =" in sql
    assert "api_access_tokens.expires_at >" in sql
    assert "RETURNING transformer.api_access_tokens.token_id" in sql
    assert hashlib.sha256(issued.token.encode("ascii")).hexdigest() in (
        compiled.params.values()
    )
    assert now in compiled.params.values()


def test_revoked_cached_token_is_rechecked_at_sixty_seconds():
    token = "a." + "B" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = _CredentialStore({
        digest: _identity(now + timedelta(hours=1)),
    })
    cache = AccessTokenCache(store)

    cache.authenticate(token, now=now, monotonic_now=10.0)
    del store.credentials[digest]

    assert cache.authenticate(
        token,
        now=now + timedelta(seconds=59, microseconds=999_000),
        monotonic_now=69.999,
    ).owner_subject == "inventory"
    with pytest.raises(InvalidAccessTokenError):
        cache.authenticate(
            token,
            now=now + timedelta(seconds=60),
            monotonic_now=70.0,
        )
    assert [lookup[0] for lookup in store.lookups] == [digest, digest]


def test_cache_does_not_store_negative_lookup():
    token = "a." + "C" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = _CredentialStore()
    cache = AccessTokenCache(store)

    with pytest.raises(InvalidAccessTokenError):
        cache.authenticate(token, now=now, monotonic_now=20.0)

    store.credentials[digest] = _identity(now + timedelta(hours=1))
    assert cache.authenticate(
        token,
        now=now + timedelta(milliseconds=1),
        monotonic_now=20.001,
    ).owner_subject == "inventory"
    assert [lookup[0] for lookup in store.lookups] == [digest, digest]


def test_new_cache_instance_starts_empty():
    token = "a." + "I" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = _CredentialStore({
        digest: _identity(now + timedelta(hours=1)),
    })

    AccessTokenCache(store).authenticate(
        token,
        now=now,
        monotonic_now=1.0,
    )
    AccessTokenCache(store).authenticate(
        token,
        now=now + timedelta(seconds=1),
        monotonic_now=2.0,
    )

    assert [lookup[0] for lookup in store.lookups] == [digest, digest]


@pytest.mark.parametrize("known", [True, False])
def test_parallel_cache_misses_share_one_postgresql_recheck(
    monkeypatch,
    known,
):
    token = "a." + "H" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    identity = _identity(now + timedelta(hours=1)) if known else None
    lookup_started = threading.Event()
    waiter_started = threading.Event()
    release_lookup = threading.Event()
    lookups = []
    outcomes = queue.Queue()

    class Store:
        def use_active_credential(self, token_digest, *, now):
            lookups.append((token_digest, now))
            lookup_started.set()
            assert release_lookup.wait(timeout=1.0)
            return identity

    cache = AccessTokenCache(Store())
    original_future_result = token_cache_module.Future.result

    def tracked_future_result(future, timeout=None):
        waiter_started.set()
        return original_future_result(future, timeout)

    def authenticate():
        try:
            outcomes.put(
                cache.authenticate(
                    token,
                    now=now,
                    monotonic_now=10.0,
                )
            )
        except BaseException as exc:
            outcomes.put(exc)

    monkeypatch.setattr(
        token_cache_module.Future,
        "result",
        tracked_future_result,
    )
    leader = threading.Thread(target=authenticate)
    follower = threading.Thread(target=authenticate)
    leader.start()
    assert lookup_started.wait(timeout=1.0)
    follower.start()
    assert waiter_started.wait(timeout=1.0)
    release_lookup.set()
    leader.join(timeout=2.0)
    follower.join(timeout=2.0)

    assert not leader.is_alive()
    assert not follower.is_alive()
    assert [lookup[0] for lookup in lookups] == [digest]
    results = [outcomes.get_nowait(), outcomes.get_nowait()]
    if known:
        assert all(
            result.owner_subject == "inventory"
            for result in results
        )
    else:
        assert all(
            isinstance(result, InvalidAccessTokenError)
            for result in results
        )
        with pytest.raises(InvalidAccessTokenError):
            cache.authenticate(
                token,
                now=now,
                monotonic_now=10.1,
            )
        assert [lookup[0] for lookup in lookups] == [digest, digest]


def test_cache_never_extends_token_expiration():
    token = "a." + "D" * 86
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = _CredentialStore({
        digest: _identity(now + timedelta(seconds=5)),
    })
    cache = AccessTokenCache(store)

    cache.authenticate(token, now=now, monotonic_now=30.0)
    with pytest.raises(InvalidAccessTokenError):
        cache.authenticate(
            token,
            now=now + timedelta(seconds=5),
            monotonic_now=35.0,
        )
    assert [lookup[0] for lookup in store.lookups] == [digest, digest]


def test_cache_evicts_least_recently_used_entry_at_capacity():
    tokens = tuple("a." + character * 86 for character in "EFG")
    digests = tuple(
        hashlib.sha256(token.encode("ascii")).hexdigest()
        for token in tokens
    )
    now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)
    store = _CredentialStore({
        digest: _identity(
            now + timedelta(hours=1),
            token_id=f"token-{index}",
        )
        for index, digest in enumerate(digests)
    })
    cache = AccessTokenCache(store, max_entries=2)

    cache.authenticate(tokens[0], now=now, monotonic_now=40.0)
    cache.authenticate(tokens[1], now=now, monotonic_now=41.0)
    cache.authenticate(tokens[0], now=now, monotonic_now=42.0)
    cache.authenticate(tokens[2], now=now, monotonic_now=43.0)
    cache.authenticate(tokens[1], now=now, monotonic_now=44.0)

    assert [lookup[0] for lookup in store.lookups] == [
        digests[0],
        digests[1],
        digests[2],
        digests[1],
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ttl_seconds": 0}, "TTL"),
        ({"ttl_seconds": math.inf}, "TTL"),
        ({"max_entries": 0}, "capacity"),
    ],
)
def test_cache_rejects_invalid_bounds(overrides, message):
    with pytest.raises(ValueError, match=message):
        AccessTokenCache(_CredentialStore(), **overrides)


def test_access_token_model_contains_digest_only():
    columns = set(ApiAccessToken.__table__.columns.keys())

    assert columns == {
        "token_id",
        "token_digest",
        "subject",
        "created_at",
        "last_used_at",
        "expires_at",
    }
    assert ApiAccessToken.__table__.c.created_at.type.timezone is True
    assert ApiAccessToken.__table__.c.last_used_at.type.timezone is True
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
