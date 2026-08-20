"""Cut over durable state to the unified Flight v5 indicator identity.

Revision ID: 0011
Revises: 0010
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    active_models = op.get_bind().scalar(sa.text(
        f"SELECT count(*) FROM {quoted}.models "
        "WHERE lifecycle_state <> 'DELETED'"
    ))
    if active_models:
        raise RuntimeError(
            "Flight v5 cutover requires every published model to be DELETED; "
            "delete models with 'transformer models delete' and let "
            "maintenance finish before applying migrations"
        )

    # Flight v4 jobs, recovery state, idempotency results and telemetry contain
    # the former semantic identities and cannot be replayed as Flight v5.
    op.execute(sa.text(f"DELETE FROM {quoted}.metrics_outbox"))
    op.execute(sa.text(f"DELETE FROM {quoted}.fit_run_summary_artifacts"))
    op.execute(sa.text(f"DELETE FROM {quoted}.training_metrics_artifacts"))
    op.execute(sa.text(f"DELETE FROM {quoted}.model_aliases"))
    op.execute(sa.text(f"UPDATE {quoted}.models SET producing_job_id = NULL"))
    op.execute(sa.text(f"DELETE FROM {quoted}.idempotency_records"))
    op.execute(sa.text(f"DELETE FROM {quoted}.jobs"))
    op.execute(sa.text(f"DELETE FROM {quoted}.job_identities"))
    op.execute(sa.text(
        f"DELETE FROM {quoted}.runtime_state WHERE key = 'storage_epoch'"
    ))
    op.execute(sa.text(
        f"ALTER SEQUENCE {quoted}.job_queue_sequence_seq RESTART WITH 1"
    ))


def downgrade() -> None:
    raise RuntimeError(
        "Flight v5 identity cutover is destructive and cannot be downgraded"
    )
