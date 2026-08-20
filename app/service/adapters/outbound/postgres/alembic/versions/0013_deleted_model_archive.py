"""Add a minimal audit archive for future model deletions.

Revision ID: 0013
Revises: 0012
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.create_table(
        "deleted_models",
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "deletion_requested_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_ref", name="deleted_models_pk"),
        sa.UniqueConstraint(
            "owner_subject",
            "label",
            "generation",
            name="deleted_models_generation_uq",
        ),
        sa.CheckConstraint(
            "generation > 0",
            name="deleted_models_generation_ck",
        ),
        schema=schema,
    )
    op.create_index(
        "deleted_models_deleted_at_idx",
        "deleted_models",
        ["deleted_at", "model_ref"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    used = op.get_bind().scalar(sa.text(
        f"SELECT EXISTS (SELECT 1 FROM {quoted}.deleted_models)"
    ))
    if used:
        raise RuntimeError(
            "deleted model audit migration cannot be downgraded after use"
        )
    op.drop_index(
        "deleted_models_deleted_at_idx",
        table_name="deleted_models",
        schema=schema,
    )
    op.drop_table("deleted_models", schema=schema)
