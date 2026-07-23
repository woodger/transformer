from __future__ import annotations

import time
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.schema import DropSchema

from app.database.migrations import (
    apply_migrations,
    migration_status,
    rollback_migration,
)
from app.database.tokens import AccessTokenStore
from app.flight.ledger import Ledger
from app.flight.token_cache import AccessTokenCache, AccessTokenCacheService


class _SilentLogger:
    def event(self, _event, **_fields):
        pass


def _wait_until(predicate, *, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition was not satisfied before timeout")


def _fit_job(ledger, label):
    job_id = str(uuid.uuid4())
    return ledger.create_job(
        job_id=job_id,
        owner_subject="inventory",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        model_label=label,
    )


def _run(ledger, job):
    ledger.seal_job(job["job_id"], manifest_hash="b" * 64, manifest=[])
    ledger.queue_job(job["job_id"], selected_device="cpu")
    return ledger.claim_next_job("cpu")


def test_postgresql_schema_is_at_alembic_head(postgres_config):
    status = migration_status(postgres_config)

    assert status.current == ("0001",)
    assert status.heads == ("0001",)
    assert status.pending is False


def test_alembic_upgrade_and_single_revision_rollback(postgres_config):
    schema = f"transformer_migration_test_{uuid.uuid4().hex}"
    config = type(postgres_config)(
        postgres_config.host,
        postgres_config.database,
        postgres_config.user,
        postgres_config.password,
        postgres_config.port,
        schema,
    )
    cleanup_engine = create_engine(postgres_config.url)
    try:
        initial = migration_status(config)
        applied = apply_migrations(config)
        rolled_back = rollback_migration(config)

        assert initial.current == ()
        assert initial.heads == ("0001",)
        assert initial.pending is True
        assert applied.current == ("0001",)
        assert applied.pending is False
        assert rolled_back.current == ()
        assert rolled_back.pending is True
    finally:
        with cleanup_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        cleanup_engine.dispose()


def test_access_tokens_use_required_format_and_list_omits_credentials(
    postgres_database,
):
    store = AccessTokenStore(postgres_database)

    issued = store.issue("inventory")
    listed = store.list()
    revoked = store.revoke(issued.token_id)

    assert issued.token.startswith("a.")
    assert len(issued.token) == 88
    assert set(issued.token[2:]) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    assert listed[0].token is None
    assert listed[0].subject == "inventory"
    assert revoked.revoked_at is not None
    assert store.active_credentials() == []


def test_access_token_cache_refreshes_after_issue_and_revoke(
    postgres_config,
    postgres_database,
):
    store = AccessTokenStore(postgres_database)
    cache = AccessTokenCache()
    listener = AccessTokenCacheService(
        postgres_config,
        store,
        cache,
        logger=_SilentLogger(),
    ).start()
    try:
        issued = store.issue("inventory")
        _wait_until(lambda: cache.lookup(issued.token) is not None)
        identity = cache.lookup(issued.token)

        assert identity.token_id == issued.token_id
        assert identity.subject == "inventory"

        store.revoke(issued.token_id)
        _wait_until(lambda: cache.lookup(issued.token) is None)
    finally:
        listener.shutdown(timeout=3.0)


def test_runtime_epoch_reset_discards_jobs_but_preserves_tokens_and_models(
    postgres_database,
):
    ledger = Ledger(postgres_database).initialize()
    first_epoch = str(uuid.uuid4())
    ledger.synchronize_runtime_epoch(first_epoch)
    token_store = AccessTokenStore(postgres_database)
    token = token_store.issue("inventory")

    producer = _fit_job(ledger, "daily")
    running = _run(ledger, producer)
    model_ref = f"mdl_{uuid.uuid4().hex}"
    ledger.publish_model(
        producer["job_id"],
        running["attempt"],
        model_ref=model_ref,
        label="daily",
        generation=None,
        checkpoint_path=f"{model_ref}/checkpoint.pth",
        metadata_path=f"{model_ref}/metadata.json",
        sha256="c" * 64,
        metadata={"modelRef": model_ref},
        result={"modelRef": model_ref},
    )
    queued = _fit_job(ledger, "next")
    ledger.seal_job(queued["job_id"], manifest_hash="d" * 64, manifest=[])
    ledger.queue_job(queued["job_id"], selected_device="cpu")

    reset = ledger.synchronize_runtime_epoch(str(uuid.uuid4()))

    assert reset["reset"] is True
    assert set(reset["discarded_jobs"]) == {producer["job_id"], queued["job_id"]}
    assert ledger.list_jobs() == []
    model = ledger.get_model(model_ref, owner_subject="inventory")
    assert model is not None
    assert model["producing_job_id"] is None
    assert ledger.resolve_model_alias("inventory", "daily")["model_ref"] == model_ref
    assert token_store.list()[0].token_id == token.token_id
    assert token_store.active_credentials()[0][0] == token.token


def test_runtime_epoch_change_is_reported_even_without_jobs(postgres_database):
    ledger = Ledger(postgres_database).initialize()

    initialized = ledger.synchronize_runtime_epoch(str(uuid.uuid4()))
    reset = ledger.synchronize_runtime_epoch(str(uuid.uuid4()))

    assert initialized == {"reset": False, "discarded_jobs": []}
    assert reset == {"reset": True, "discarded_jobs": []}
