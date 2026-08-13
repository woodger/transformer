from __future__ import annotations

import time
import uuid

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.schema import DropSchema

from app.contracts.worker.v3.config import TrainConfig
from app.contracts.worker.v3.objective import ml_contract
from app.service.adapters.inbound.flight.constants import FIT_SCHEMA_ID
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.migrations import (
    alembic_config,
    apply_migrations,
    migration_status,
    rollback_migration,
)
from app.service.adapters.outbound.postgres.token_cache import (
    AccessTokenCache,
    AccessTokenCacheService,
)
from app.service.adapters.outbound.postgres.tokens import AccessTokenStore


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
        client_execution_id=str(uuid.uuid4()),
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        data_contract=_data_contract(),
        ml_contract=ml_contract(TrainConfig()),
        create_result={"jobId": job_id},
        model_label=label,
        model_config=_model_config(),
        training_config={},
    )


def _data_contract():
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "data_contract_sha256": "d" * 64,
        "seq_len": 2,
        "feature_dim": 1,
        "target_schema_id": "inventory.target.v1",
    }


def _model_config():
    return {
        "seq_len": 2,
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
        "nhead": 2,
        "context_mode": "relaxed",
        "out_dim": 6,
        "feature_dim": 1,
    }


def _commit_input(ledger, job, *, storage_class):
    payload_id = str(uuid.uuid4())
    upload_token = uuid.uuid4().hex
    relative_path = (
        f"jobs/{job['job_id']}/inputs/0-{payload_id}.arrow"
    )
    ledger.reserve_input(
        job_id=job["job_id"],
        payload_id=payload_id,
        ordinal=0,
        client_execution_id=job["client_execution_id"],
        fencing_token=job["fencing_token"],
        upload_token=upload_token,
        candidate_path=relative_path,
        storage_class=storage_class,
    )
    ledger.commit_input(
        upload_token=upload_token,
        job_id=job["job_id"],
        client_execution_id=job["client_execution_id"],
        fencing_token=job["fencing_token"],
        relative_path=relative_path,
        schema_id=FIT_SCHEMA_ID,
        data_contract_sha256="d" * 64,
        rows=1,
        batches=1,
        byte_count=1,
        sha256="1" * 64,
        schema_fingerprint="2" * 64,
        source_width=2,
        feature_dim=1,
        selected_device="cpu",
        max_payloads=10,
        max_job_bytes=10,
        storage_class=storage_class,
    )


def test_postgresql_schema_is_at_alembic_head(postgres_config):
    status = migration_status(postgres_config)

    assert status.current == ("0006",)
    assert status.heads == ("0006",)
    assert status.pending is False


def test_v4_schema_migration_is_irreversible(
    postgres_config,
):
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
        with pytest.raises(RuntimeError, match="cannot be downgraded"):
            rollback_migration(config)
        after_failed_rollback = migration_status(config)

        assert initial.current == ()
        assert initial.heads == ("0006",)
        assert initial.pending is True
        assert applied.current == ("0006",)
        assert applied.pending is False
        assert after_failed_rollback.current == ("0006",)
        assert after_failed_rollback.pending is False
    finally:
        with cleanup_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        cleanup_engine.dispose()


def test_v4_schema_migration_preserves_tokens_and_model_identities_only(
    postgres_config,
):
    schema = f"transformer_v4_schema_test_{uuid.uuid4().hex}"
    config = type(postgres_config)(
        postgres_config.host,
        postgres_config.database,
        postgres_config.user,
        postgres_config.password,
        postgres_config.port,
        schema,
    )
    engine = create_engine(config.url)
    job_id = str(uuid.uuid4())
    token_id = str(uuid.uuid4())
    model_ref = "mdl_preserved"
    quoted = f'"{schema}"'
    try:
        command.upgrade(alembic_config(config), "0003")
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {quoted}.jobs (
                        job_id, owner_subject, operation, state, revision,
                        requested_device, model_label, prediction_column,
                        config_hash, created_at, updated_at
                    ) VALUES (
                        :job_id, 'inventory', 'fit', 'UPLOADING', 1,
                        'cpu', 'daily', 'out', :config_hash, now(), now()
                    )
                    """
                ),
                {"job_id": job_id, "config_hash": "c" * 64},
            )
            connection.execute(
                text(
                    f"""
                    INSERT INTO {quoted}.models (
                        model_ref, owner_subject, label, generation,
                        checkpoint_path, metadata_path, sha256, metadata,
                        producing_job_id, created_at
                    ) VALUES (
                        :model_ref, 'inventory', 'daily', 1,
                        'models/daily/1/checkpoint.pth',
                        'models/daily/1/metadata.json', :sha256,
                        '{{}}'::jsonb, :job_id, now()
                    )
                    """
                ),
                {
                    "model_ref": model_ref,
                    "sha256": "m" * 64,
                    "job_id": job_id,
                },
            )
            connection.execute(
                text(
                    f"""
                    INSERT INTO {quoted}.model_aliases (
                        owner_subject, label, model_ref, updated_at
                    ) VALUES ('inventory', 'daily', :model_ref, now())
                    """
                ),
                {"model_ref": model_ref},
            )
            connection.execute(
                text(
                    f"""
                    INSERT INTO {quoted}.api_access_tokens (
                        token_id, token, subject, created_at
                    ) VALUES (:token_id, :token, 'inventory', now())
                    """
                ),
                {"token_id": token_id, "token": "a." + "A" * 86},
            )
            connection.execute(
                text(
                    f"""
                    INSERT INTO {quoted}.idempotency_records (
                        owner_subject, action_name, idempotency_key,
                        request_hash, response, job_id, created_at
                    ) VALUES (
                        'inventory', 'transformer.v2.job.create', 'create:1',
                        :request_hash, '{{}}'::jsonb, :job_id, now()
                    )
                    """
                ),
                {"request_hash": "i" * 64, "job_id": job_id},
            )
            connection.execute(
                text(
                    f"""
                    INSERT INTO {quoted}.runtime_state (key, value, updated_at)
                    VALUES ('storage_epoch', 'v2-runtime', now())
                    """
                )
            )

        command.upgrade(alembic_config(config), "head")

        with engine.connect() as connection:
            jobs = connection.scalar(text(f"SELECT count(*) FROM {quoted}.jobs"))
            identities = connection.scalar(
                text(f"SELECT count(*) FROM {quoted}.job_identities")
            )
            idempotency = connection.scalar(
                text(f"SELECT count(*) FROM {quoted}.idempotency_records")
            )
            storage_epoch = connection.scalar(
                text(
                    f"SELECT count(*) FROM {quoted}.runtime_state "
                    "WHERE key = 'storage_epoch'"
                )
            )
            token = connection.execute(
                text(
                    f"SELECT token_id, subject FROM {quoted}.api_access_tokens"
                )
            ).one()
            model = connection.execute(
                text(
                    f"""
                    SELECT producing_job_id, checkpoint_bytes,
                           data_contract, data_contract_sha256,
                           ml_contract, objective_config_sha256
                    FROM {quoted}.models
                    WHERE model_ref = :model_ref
                    """
                ),
                {"model_ref": model_ref},
            ).one()
            alias = connection.scalar(
                text(
                    f"""
                    SELECT model_ref FROM {quoted}.model_aliases
                    WHERE owner_subject = 'inventory' AND label = 'daily'
                    """
                )
            )
            model_ref_lengths = connection.execute(
                text(
                    """
                    SELECT table_name, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND column_name = 'model_ref'
                      AND table_name IN ('models', 'model_aliases')
                    ORDER BY table_name
                    """
                ),
                {"schema": schema},
            ).all()

        assert jobs == 0
        assert identities == 0
        assert idempotency == 0
        assert storage_epoch == 0
        assert token == (uuid.UUID(token_id), "inventory")
        assert model == (None, 1, None, None, None, None)
        assert alias == model_ref
        assert model_ref_lengths == [
            ("model_aliases", 128),
            ("models", 128),
        ]
        assert migration_status(config).current == ("0006",)
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        engine.dispose()


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


def test_runtime_epoch_reset_discards_runtime_jobs_and_preserves_recovery_fits(
    postgres_database,
):
    ledger = Ledger(postgres_database).initialize()
    first_epoch = str(uuid.uuid4())
    ledger.synchronize_runtime_epoch(first_epoch)
    token_store = AccessTokenStore(postgres_database)
    token = token_store.issue("inventory")

    persistent_fit = _fit_job(ledger, "daily")
    _commit_input(ledger, persistent_fit, storage_class="recovery")
    prediction = ledger.create_job(
        job_id=str(uuid.uuid4()),
        owner_subject="inventory",
        client_execution_id=str(uuid.uuid4()),
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="e" * 64,
        data_contract=_data_contract(),
        ml_contract=ml_contract(TrainConfig()),
        create_result={"jobId": "predict"},
        resolved_model_ref="mdl_seed",
        model_config=_model_config(),
    )
    runtime_fit = _fit_job(ledger, "runtime")
    _commit_input(ledger, runtime_fit, storage_class="runtime")

    reset = ledger.synchronize_runtime_epoch(str(uuid.uuid4()))

    assert reset["reset"] is True
    assert set(reset["discarded_jobs"]) == {
        prediction["job_id"],
        runtime_fit["job_id"],
    }
    assert {
        job["job_id"]
        for job in ledger.list_jobs()
    } == {persistent_fit["job_id"]}
    assert ledger.get_job_identity(prediction["job_id"])["retired_at"] is not None
    assert ledger.get_job_identity(runtime_fit["job_id"])["retired_at"] is not None
    assert token_store.list()[0].token_id == token.token_id
    assert token_store.active_credentials()[0][0] == token.token


def test_runtime_epoch_change_is_reported_even_without_jobs(postgres_database):
    ledger = Ledger(postgres_database).initialize()

    initialized = ledger.synchronize_runtime_epoch(str(uuid.uuid4()))
    reset = ledger.synchronize_runtime_epoch(str(uuid.uuid4()))

    assert initialized == {"reset": False, "discarded_jobs": []}
    assert reset == {"reset": True, "discarded_jobs": []}
