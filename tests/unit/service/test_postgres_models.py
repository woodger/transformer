from __future__ import annotations

from typing import get_type_hints

from app.service.adapters.outbound.postgres.models import Base


def test_all_postgres_mapped_annotations_resolve_at_runtime():
    for mapper in Base.registry.mappers:
        get_type_hints(mapper.class_, include_extras=True)
