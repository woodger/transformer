"""Remove standalone published model metadata sidecars.

Revision ID: 0022
Revises: 0021
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.drop_constraint(
        "models_metadata_path_uq",
        "models",
        schema=schema,
        type_="unique",
    )
    op.drop_column("models", "metadata_path", schema=schema)


def downgrade() -> None:
    schema = _schema()
    op.add_column(
        "models",
        sa.Column("metadata_path", sa.Text()),
        schema=schema,
    )
    models = sa.table(
        "models",
        sa.column("model_ref", sa.String(128)),
        sa.column("metadata_path", sa.Text()),
        schema=schema,
    )
    op.execute(
        models.update().values(
            metadata_path=models.c.model_ref + "/metadata.json"
        )
    )
    op.alter_column(
        "models",
        "metadata_path",
        existing_type=sa.Text(),
        nullable=False,
        schema=schema,
    )
    op.create_unique_constraint(
        "models_metadata_path_uq",
        "models",
        ["metadata_path"],
        schema=schema,
    )
