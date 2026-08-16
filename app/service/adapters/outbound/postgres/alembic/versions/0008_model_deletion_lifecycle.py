"""Add the durable published-model deletion lifecycle.

Revision ID: 0008
Revises: 0007
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "models",
        sa.Column(
            "lifecycle_state",
            sa.String(16),
            nullable=False,
            server_default="AVAILABLE",
        ),
        schema=schema,
    )
    op.add_column(
        "models",
        sa.Column(
            "deletion_requested_at",
            sa.DateTime(timezone=True),
        ),
        schema=schema,
    )
    op.add_column(
        "models",
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    op.create_check_constraint(
        "models_lifecycle_ck",
        "models",
        "(lifecycle_state = 'AVAILABLE' "
        "AND deletion_requested_at IS NULL AND deleted_at IS NULL) OR "
        "(lifecycle_state = 'DELETING' "
        "AND deletion_requested_at IS NOT NULL AND deleted_at IS NULL) OR "
        "(lifecycle_state = 'DELETED' "
        "AND deletion_requested_at IS NOT NULL AND deleted_at IS NOT NULL)",
        schema=schema,
    )
    op.create_index(
        "models_deleting_idx",
        "models",
        ["deletion_requested_at", "model_ref"],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("lifecycle_state = 'DELETING'"),
    )
    op.drop_constraint(
        "metrics_outbox_status_ck",
        "metrics_outbox",
        schema=schema,
        type_="check",
    )
    op.create_check_constraint(
        "metrics_outbox_status_ck",
        "metrics_outbox",
        "status IN ('PENDING', 'BLOCKED', 'DELIVERED', 'CANCELLED')",
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    quoted_schema = op.get_bind().dialect.identifier_preparer.quote(schema)
    used = op.get_bind().scalar(sa.text(
        f"SELECT EXISTS ("
        f"SELECT 1 FROM {quoted_schema}.models "
        "WHERE lifecycle_state <> 'AVAILABLE' "
        "UNION ALL "
        f"SELECT 1 FROM {quoted_schema}.metrics_outbox "
        "WHERE status = 'CANCELLED'"
        ")"
    ))
    if used:
        raise RuntimeError(
            "model deletion lifecycle migration cannot be downgraded after use"
        )

    op.drop_constraint(
        "metrics_outbox_status_ck",
        "metrics_outbox",
        schema=schema,
        type_="check",
    )
    op.create_check_constraint(
        "metrics_outbox_status_ck",
        "metrics_outbox",
        "status IN ('PENDING', 'BLOCKED', 'DELIVERED')",
        schema=schema,
    )
    op.drop_index(
        "models_deleting_idx",
        table_name="models",
        schema=schema,
    )
    op.drop_constraint(
        "models_lifecycle_ck",
        "models",
        schema=schema,
        type_="check",
    )
    op.drop_column("models", "deleted_at", schema=schema)
    op.drop_column("models", "deletion_requested_at", schema=schema)
    op.drop_column("models", "lifecycle_state", schema=schema)
