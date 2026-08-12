"""Align published model reference columns with the ORM contract.

Revision ID: 0005
Revises: 0004
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.drop_constraint(
        "model_aliases_model_ref_fkey",
        "model_aliases",
        schema=schema,
        type_="foreignkey",
    )
    op.alter_column(
        "models",
        "model_ref",
        existing_type=sa.String(64),
        type_=sa.String(128),
        existing_nullable=False,
        schema=schema,
    )
    op.alter_column(
        "model_aliases",
        "model_ref",
        existing_type=sa.String(64),
        type_=sa.String(128),
        existing_nullable=False,
        schema=schema,
    )
    op.create_foreign_key(
        "model_aliases_model_ref_fkey",
        "model_aliases",
        "models",
        ["model_ref"],
        ["model_ref"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="CASCADE",
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_constraint(
        "model_aliases_model_ref_fkey",
        "model_aliases",
        schema=schema,
        type_="foreignkey",
    )
    op.alter_column(
        "models",
        "model_ref",
        existing_type=sa.String(128),
        type_=sa.String(64),
        existing_nullable=False,
        schema=schema,
    )
    op.alter_column(
        "model_aliases",
        "model_ref",
        existing_type=sa.String(128),
        type_=sa.String(64),
        existing_nullable=False,
        schema=schema,
    )
    op.create_foreign_key(
        "model_aliases_model_ref_fkey",
        "model_aliases",
        "models",
        ["model_ref"],
        ["model_ref"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="CASCADE",
    )
