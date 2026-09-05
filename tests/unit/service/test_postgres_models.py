from __future__ import annotations

from typing import get_type_hints

from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

from app.service.adapters.outbound.postgres.models import Base, Job


def test_all_postgres_mapped_annotations_resolve_at_runtime():
    for mapper in Base.registry.mappers:
        get_type_hints(mapper.class_, include_extras=True)


def test_absent_job_initialization_binds_as_sql_null():
    column_type = Job.__table__.c.initialization.type

    assert isinstance(column_type, JSONB)
    assert column_type.none_as_null is True
    processor = column_type.bind_processor(postgresql.dialect())
    assert processor is not None
    assert processor(None) is None
