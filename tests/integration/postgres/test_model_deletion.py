from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.models import (
    FitRunSummaryArtifact,
    MetricsOutboxEntry,
    ModelAlias,
    PublishedModel,
    TrainingMetricsArtifact,
)
from app.service.adapters.outbound.postgres.published_models import (
    PublishedModelStore,
)
from app.service.adapters.outbound.postgres.telemetry import (
    PostgresMetricsOutbox,
)
from app.service.application.services.maintenance import MaintenanceService
from app.service.domain.job import ExecutionState
from app.service.domain.model import (
    ModelDeletionBlocked,
    ModelLifecycleState,
)
from tests.support.flight_v4_helpers import (
    close_input,
    commit_input,
    create_fit,
    create_predict,
    internal_data_contract,
    model_config,
)


def _seed_model(
    database,
    model_ref: str,
    *,
    generation: int = 1,
    alias: bool = True,
    outbox_status: str | None = None,
) -> str | None:
    now = datetime.now(UTC)
    with database.transaction() as session:
        session.add(PublishedModel(
            model_ref=model_ref,
            owner_subject="inventory",
            label="daily",
            generation=generation,
            checkpoint_path=f"{model_ref}/checkpoint.pth",
            metadata_path=f"{model_ref}/metadata.json",
            checkpoint_bytes=10,
            sha256="a" * 64,
            metadata_json={"modelRef": model_ref},
            producing_job_id=None,
            created_at=now,
        ))
        session.flush()
        if alias:
            session.add(ModelAlias(
                owner_subject="inventory",
                label="daily",
                model_ref=model_ref,
                updated_at=now,
            ))
        if outbox_status is not None:
            job_id = str(uuid.uuid4())
            attempt_id = str(uuid.uuid4())
            session.add(TrainingMetricsArtifact(
                job_id=job_id,
                model_ref=model_ref,
                format="transformer.training-metrics.v2",
                media_type="application/x-ndjson",
                relative_path=f"{job_id}/metrics.jsonl",
                bytes=10,
                sha256="b" * 64,
                row_count=1,
                attempt_id=attempt_id,
                attempt=1,
                application_version="0.1.12",
                git_commit="0" * 40,
                created_at=now,
            ))
            session.add(FitRunSummaryArtifact(
                job_id=job_id,
                model_ref=model_ref,
                format="transformer.fit-run-summary.v2",
                media_type="application/json",
                relative_path=f"{job_id}/run-summary.json",
                bytes=10,
                sha256="c" * 64,
                attempt_id=attempt_id,
                attempt=1,
                application_version="0.1.12",
                git_commit="0" * 40,
                created_at=now,
            ))
            session.flush()
            session.add(MetricsOutboxEntry(
                job_id=job_id,
                projection_version="inventory.metrics.v3",
                status=outbox_status,
                cursor=0,
                attempts=0,
                next_attempt_at=now,
                created_at=now,
                updated_at=now,
                delivered_at=(
                    now if outbox_status == "DELIVERED" else None
                ),
            ))
            return job_id
    return None


def test_model_deletion_keeps_run_telemetry_and_model_tombstone(
    postgres_database,
):
    model_ref = f"mdl_{uuid.uuid4().hex}"
    telemetry_job_id = _seed_model(
        postgres_database,
        model_ref,
        outbox_status="PENDING",
    )
    assert telemetry_job_id is not None
    store = PublishedModelStore(postgres_database)
    ledger = Ledger(postgres_database).initialize()

    requested = store.request_deletion(model_ref)
    repeated = store.request_deletion(model_ref)

    assert requested.state == ModelLifecycleState.DELETING
    pending = PostgresMetricsOutbox(postgres_database).next_pending()
    assert pending is not None
    assert pending.training_metrics.job_id == telemetry_job_id
    assert repeated.state == ModelLifecycleState.DELETING
    assert ledger.get_model(model_ref) is None
    assert ledger.resolve_model_alias("inventory", "daily") is None
    assert store.retained_model_refs() == {model_ref}
    backlog_entries, backlog_bytes, _oldest_age = (
        PostgresMetricsOutbox(postgres_database).backlog()
    )
    assert (backlog_entries, backlog_bytes) == (1, 20)

    assert store.complete_deletion(model_ref) is True
    assert store.complete_deletion(model_ref) is False
    assert store.retained_model_refs() == set()
    listed = store.list_models()
    assert len(listed) == 1
    assert listed[0].state == ModelLifecycleState.DELETED
    assert listed[0].deleted_at is not None

    with postgres_database.session() as session:
        assert session.get(PublishedModel, model_ref) is not None
        assert session.get(TrainingMetricsArtifact, telemetry_job_id) is not None
        assert session.get(FitRunSummaryArtifact, telemetry_job_id) is not None
        assert session.get(MetricsOutboxEntry, telemetry_job_id) is not None


def test_terminal_telemetry_retention_does_not_change_the_model(
    postgres_database,
):
    model_ref = f"mdl_{uuid.uuid4().hex}"
    job_id = _seed_model(
        postgres_database,
        model_ref,
        outbox_status="DELIVERED",
    )
    assert job_id is not None

    cleanups = PostgresMetricsOutbox(postgres_database).purge_terminal(
        older_than_seconds=0,
    )

    assert len(cleanups) == 1
    assert cleanups[0].job_id == job_id
    assert cleanups[0].relative_paths == (
        f"{job_id}/metrics.jsonl",
        f"{job_id}/run-summary.json",
    )
    with postgres_database.session() as session:
        model = session.get(PublishedModel, model_ref)
        assert model is not None
        assert model.lifecycle_state == ModelLifecycleState.AVAILABLE.value
        assert session.get(TrainingMetricsArtifact, job_id) is None
        assert session.get(FitRunSummaryArtifact, job_id) is None
        assert session.get(MetricsOutboxEntry, job_id) is None


def test_model_deletion_is_blocked_by_active_predict_job(
    postgres_database,
):
    model_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, model_ref)
    ledger = Ledger(postgres_database).initialize()
    job = create_predict(ledger, model_ref=model_ref)
    store = PublishedModelStore(postgres_database)

    with pytest.raises(ModelDeletionBlocked, match="active predict job"):
        store.request_deletion(model_ref)

    ledger.transition_job(
        job["job_id"],
        ExecutionState.CANCELLED,
        updates={"finished_at": datetime.now(UTC).timestamp()},
    )
    requested = store.request_deletion(model_ref)

    assert requested.state == ModelLifecycleState.DELETING


def test_deleting_non_current_generation_does_not_change_alias(
    postgres_database,
):
    old_ref = f"mdl_{uuid.uuid4().hex}"
    current_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, old_ref, generation=1, alias=False)
    _seed_model(postgres_database, current_ref, generation=2, alias=True)
    store = PublishedModelStore(postgres_database)

    store.request_deletion(old_ref)

    with postgres_database.session() as session:
        alias = session.scalar(select(ModelAlias))
        assert alias is not None
        assert alias.model_ref == current_ref

    store.request_deletion(current_ref)

    with postgres_database.session() as session:
        assert session.scalar(select(ModelAlias)) is None


def test_maintenance_removes_model_directory_and_finalizes_database_state(
    tmp_path,
    postgres_database,
):
    model_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, model_ref)
    store = PublishedModelStore(postgres_database)
    ledger = Ledger(postgres_database).initialize()
    spool = Spool(
        str(tmp_path / "runtime"),
        str(tmp_path / "models"),
    ).initialize()
    model_directory = Path(spool.model_directory(model_ref))
    model_directory.mkdir()
    (model_directory / "checkpoint.pth").write_bytes(b"checkpoint")
    store.request_deletion(model_ref)
    maintenance = MaintenanceService(
        SimpleNamespace(
            retention_seconds=60,
            input_idle_timeout_seconds=60,
        ),
        ledger,
        spool,
        model_deletions=store,
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        metrics=OperationalMetrics(),
    )

    result = maintenance.run_once(now=datetime.now(UTC).timestamp())

    assert result.completed_model_deletions == (model_ref,)
    assert not model_directory.exists()
    assert store.list_models()[0].state == ModelLifecycleState.DELETED


def test_deleted_tombstone_keeps_next_generation_monotonic(
    tmp_path,
    postgres_database,
):
    deleted_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, deleted_ref, generation=1)
    store = PublishedModelStore(postgres_database)
    store.request_deletion(deleted_ref)
    store.complete_deletion(deleted_ref)

    ledger = Ledger(postgres_database).initialize()
    spool = Spool(
        str(tmp_path / "runtime"),
        str(tmp_path / "models"),
    ).initialize()
    job = create_fit(ledger)
    commit_input(ledger, job, 0)
    close_input(ledger, job)
    attempt = ledger.claim_execution_job(job["job_id"], "cpu")
    assert attempt is not None
    model_ref = f"mdl_{uuid.uuid4().hex}"
    checkpoint_path = spool.model_checkpoint_path(model_ref)
    metadata_path = spool.model_metadata_path(model_ref)
    checkpoint = b"checkpoint"
    spool.atomic_write_bytes(checkpoint_path, checkpoint)
    spool.atomic_write_json(metadata_path, {"modelRef": model_ref})
    ledger.publish_model(
        job["job_id"],
        attempt.attempt,
        attempt_id=attempt.attempt_id,
        model_ref=model_ref,
        label="daily",
        generation=None,
        checkpoint_path=spool.model_relative_path(checkpoint_path),
        metadata_path=spool.model_relative_path(metadata_path),
        byte_count=len(checkpoint),
        sha256=hashlib.sha256(checkpoint).hexdigest(),
        metadata={
            "model_config": model_config().to_dict(),
            "data_contract": internal_data_contract(),
        },
        result={"modelRef": model_ref},
    )

    published = ledger.get_model(model_ref)
    assert published is not None
    assert published["generation"] == 2
