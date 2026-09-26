"""Добавить ограниченный идентификатор обхода каталога моделей.

Идентификатор ревизии: 0025
Предыдущая ревизия: 0024
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "models",
        sa.Column(
            "catalog_ordinal",
            sa.BigInteger(),
            sa.Identity(),
            nullable=False,
        ),
        schema=schema,
    )
    op.create_unique_constraint(
        "models_catalog_ordinal_uq",
        "models",
        ["catalog_ordinal"],
        schema=schema,
    )
    op.create_index(
        "models_catalog_owner_order_idx",
        "models",
        ["owner_subject", sa.text("created_at DESC"), "model_ref"],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("lifecycle_state = 'AVAILABLE'"),
    )
    op.create_index(
        "models_catalog_owner_high_water_idx",
        "models",
        ["owner_subject", sa.text("catalog_ordinal DESC")],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("lifecycle_state = 'AVAILABLE'"),
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "models_catalog_owner_high_water_idx",
        table_name="models",
        schema=schema,
    )
    op.drop_index(
        "models_catalog_owner_order_idx",
        table_name="models",
        schema=schema,
    )
    op.drop_constraint(
        "models_catalog_ordinal_uq",
        "models",
        schema=schema,
        type_="unique",
    )
    op.drop_column("models", "catalog_ordinal", schema=schema)
