import importlib

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.service.adapters.outbound.postgres.models import Base


def _migration():
    return importlib.import_module(
        "app.service.adapters.outbound.postgres.alembic.versions."
        "0020_current_schema_baseline"
    )


def _normalized(value):
    return " ".join(str(value).split())


def _column_contract(table):
    dialect = postgresql.dialect()
    return {
        column.name: (
            column.type.compile(dialect=dialect),
            column.nullable,
            column.primary_key,
        )
        for column in table.columns
    }


def _foreign_key_contract(table):
    return {
        (
            tuple(column.name for column in constraint.columns),
            tuple(element.target_fullname for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in table.foreign_key_constraints
    }


def _check_contract(table):
    return {
        (constraint.name, _normalized(constraint.sqltext))
        for constraint in table.constraints
        if isinstance(constraint, sa.CheckConstraint)
    }


def _unique_contract(table):
    return {
        (constraint.name, tuple(column.name for column in constraint.columns))
        for constraint in table.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    }


def _index_contract(table):
    return {
        (
            index.name,
            tuple(column.name for column in index.columns),
            index.unique,
            _normalized(index.dialect_options["postgresql"].get("where")),
        )
        for index in table.indexes
    }


def test_baseline_is_revision_0020_without_predecessors():
    migration = _migration()

    assert migration.revision == "0020"
    assert migration.down_revision is None


def test_baseline_matches_current_orm_schema_contract():
    baseline = _migration()._metadata("transformer")

    assert set(baseline.tables) == set(Base.metadata.tables)
    for key, orm_table in Base.metadata.tables.items():
        baseline_table = baseline.tables[key]
        assert _column_contract(baseline_table) == _column_contract(orm_table)
        assert _foreign_key_contract(baseline_table) == _foreign_key_contract(orm_table)
        assert _check_contract(baseline_table) == _check_contract(orm_table)
        assert _unique_contract(baseline_table) == _unique_contract(orm_table)
        assert _index_contract(baseline_table) == _index_contract(orm_table)


def test_baseline_preserves_published_server_defaults_and_column_order():
    metadata = _migration()._metadata("transformer")
    defaults = {
        f"{table.name}.{column.name}": _normalized(column.server_default.arg)
        for table in metadata.tables.values()
        for column in table.columns
        if column.server_default is not None
    }

    assert defaults == {
        "jobs.payload_count": "0",
        "jobs.total_rows": "0",
        "jobs.total_bytes": "0",
        "jobs.progress": "'{}'::jsonb",
        "jobs.attempt": "0",
        "jobs.waiting_for_input": "false",
        "models.checkpoint_bytes": "1",
        "models.lifecycle_state": "AVAILABLE",
    }
    assert list(metadata.tables["transformer.jobs"].columns)[-1].name == "ml_contract"
    assert [
        column.name for column in metadata.tables["transformer.job_attempts"].columns
    ][-3:] == [
        "queue_entered_at",
        "worker_ready_at",
        "worker_completed_at",
    ]
    assert [
        column.name
        for column in metadata.tables["transformer.training_metric_intervals"].columns
    ][-2:] == [
        "checkpoint_serialization_ms",
        "checkpoint_publication_ms",
    ]
    assert [
        column.name
        for column in metadata.tables["transformer.api_access_tokens"].columns
    ] == [
        "token_id",
        "token_digest",
        "subject",
        "created_at",
        "expires_at",
        "last_used_at",
    ]
    assert set(metadata._sequences) == {"transformer.job_queue_sequence_seq"}


def test_baseline_emits_only_current_schema_objects(monkeypatch):
    statements = []
    dialect = postgresql.dialect()
    bind = sa.create_mock_engine(
        "postgresql://",
        lambda statement, *args, **kwargs: statements.append(
            str(statement.compile(dialect=dialect))
        ),
    )
    migration = _migration()
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration, "_schema", lambda: "transformer")

    migration.upgrade()

    ddl = "\n".join(statements)
    assert ddl.count("CREATE TABLE") == 18
    assert ddl.count("CREATE SEQUENCE") == 1
    assert "CREATE TABLE transformer.api_access_tokens" in ddl
    assert "last_used_at TIMESTAMP WITH TIME ZONE" in ddl
    assert "DROP " not in ddl
    assert "ALTER " not in ddl
    assert "notify_auth_token_change" not in ddl


def test_baseline_cannot_be_downgraded():
    with pytest.raises(RuntimeError, match="schema baseline"):
        _migration().downgrade()
