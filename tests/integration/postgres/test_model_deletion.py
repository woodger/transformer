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
from app.service.adapters.outbound.postgres.metrics_outbox import (
    PostgresMetricsOutbox,
)
from app.service.adapters.outbound.postgres.models import (
    MetricsOutboxEntry,
    ModelAlias,
    ModelMetricsArtifact,
    PublishedModel,
)
from app.service.adapters.outbound.postgres.published_models import (
    PublishedModelStore,
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
    create_test_metrics_artifact,
    create_test_run_summary_artifact,
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
) -> None:
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
            session.add(ModelMetricsArtifact(
                model_ref=model_ref,
                format="transformer.training-metrics.v1",
                media_type="application/x-ndjson",
                relative_path=f"{model_ref}/metrics.jsonl",
                bytes=10,
                sha256="b" * 64,
                row_count=1,
                job_id=job_id,
                attempt_id=str(uuid.uuid4()),
                attempt=1,
                application_version="0.1.10",
                git_commit="0" * 40,
                created_at=now,
            ))
            session.flush()
            session.add(MetricsOutboxEntry(
                model_ref=model_ref,
                projection_version="inventory.metrics.v1",
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


def test_model_deletion_requires_explicit_metrics_discard_and_keeps_tombstone(
    postgres_database,
):
    model_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, model_ref, outbox_status="PENDING")
    store = PublishedModelStore(postgres_database)
    ledger = Ledger(postgres_database).initialize()

    with pytest.raises(ModelDeletionBlocked, match="undelivered training metrics"):
        store.request_deletion(
            model_ref,
            discard_undelivered_metrics=False,
        )

    assert ledger.get_model(model_ref) is not None
    assert ledger.resolve_model_alias("inventory", "daily") is not None

    requested = store.request_deletion(
        model_ref,
        discard_undelivered_metrics=True,
    )
    repeated = store.request_deletion(
        model_ref,
        discard_undelivered_metrics=False,
    )

    assert requested.state == ModelLifecycleState.DELETING
    assert requested.metrics_delivery_status == "CANCELLED"
    assert repeated.state == ModelLifecycleState.DELETING
    assert ledger.get_model(model_ref) is None
    assert ledger.resolve_model_alias("inventory", "daily") is None
    assert store.retained_model_refs() == {model_ref}
    assert PostgresMetricsOutbox(postgres_database).backlog() == (0, 0, None)

    assert store.complete_deletion(model_ref) is True
    assert store.complete_deletion(model_ref) is False
    assert store.retained_model_refs() == set()
    listed = store.list_models()
    assert len(listed) == 1
    assert listed[0].state == ModelLifecycleState.DELETED
    assert listed[0].deleted_at is not None

    with postgres_database.session() as session:
        assert session.get(PublishedModel, model_ref) is not None
        assert session.get(ModelMetricsArtifact, model_ref) is None
        assert session.get(MetricsOutboxEntry, model_ref) is None


def test_model_deletion_is_blocked_by_active_predict_job(
    postgres_database,
):
    model_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, model_ref)
    ledger = Ledger(postgres_database).initialize()
    job = create_predict(ledger, model_ref=model_ref)
    store = PublishedModelStore(postgres_database)

    with pytest.raises(ModelDeletionBlocked, match="active predict job"):
        store.request_deletion(
            model_ref,
            discard_undelivered_metrics=False,
        )

    ledger.transition_job(
        job["job_id"],
        ExecutionState.CANCELLED,
        updates={"finished_at": datetime.now(UTC).timestamp()},
    )
    requested = store.request_deletion(
        model_ref,
        discard_undelivered_metrics=False,
    )

    assert requested.state == ModelLifecycleState.DELETING


def test_deleting_non_current_generation_does_not_change_alias(
    postgres_database,
):
    old_ref = f"mdl_{uuid.uuid4().hex}"
    current_ref = f"mdl_{uuid.uuid4().hex}"
    _seed_model(postgres_database, old_ref, generation=1, alias=False)
    _seed_model(postgres_database, current_ref, generation=2, alias=True)
    store = PublishedModelStore(postgres_database)

    store.request_deletion(
        old_ref,
        discard_undelivered_metrics=False,
    )

    with postgres_database.session() as session:
        alias = session.scalar(select(ModelAlias))
        assert alias is not None
        assert alias.model_ref == current_ref

    store.request_deletion(
        current_ref,
        discard_undelivered_metrics=False,
    )

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
    store.request_deletion(
        model_ref,
        discard_undelivered_metrics=False,
    )
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
    store.request_deletion(
        deleted_ref,
        discard_undelivered_metrics=False,
    )
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
    metrics = create_test_metrics_artifact(
        spool,
        model_ref=model_ref,
        job_id=job["job_id"],
        attempt_id=attempt.attempt_id,
        attempt=attempt.attempt,
    )
    run_summary = create_test_run_summary_artifact(
        spool,
        model_ref=model_ref,
        job_id=job["job_id"],
        attempt_id=attempt.attempt_id,
        attempt=attempt.attempt,
    )

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
        metrics_path=metrics.relative_path,
        metrics_format="transformer.training-metrics.v1",
        metrics_media_type="application/x-ndjson",
        metrics_byte_count=metrics.byte_count,
        metrics_sha256=metrics.sha256,
        metrics_row_count=metrics.row_count,
        run_summary_path=run_summary.relative_path,
        run_summary_format="transformer.fit-run-summary.v1",
        run_summary_media_type="application/json",
        run_summary_byte_count=run_summary.byte_count,
        run_summary_sha256=run_summary.sha256,
        application_version="0.1.10",
        git_commit="0" * 40,
        metadata={
            "model_config": model_config().to_dict(),
            "data_contract": internal_data_contract(),
        },
        result={"modelRef": model_ref},
    )

    published = ledger.get_model(model_ref)
    assert published is not None
    assert published["generation"] == 2
