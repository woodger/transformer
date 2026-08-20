from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from app.service.adapters.outbound.postgres.models import (
    DeletedModel,
    PublishedModel,
)
from app.service.adapters.outbound.postgres.published_models import (
    PublishedModelStore,
)


class SessionDouble:
    def __init__(self, model):
        self.model = model
        self.added = []
        self.deleted = []
        self.flushed = False

    def get(self, model_type, model_ref, *, with_for_update):
        assert model_type is PublishedModel
        assert model_ref == "mdl_0123456789abcdef0123456789abcdef"
        assert with_for_update is True
        return self.model

    def delete(self, model):
        self.deleted.append(model)

    def add(self, model):
        self.added.append(model)

    def flush(self):
        self.flushed = True


class DatabaseDouble:
    def __init__(self, session):
        self.session = session

    @contextmanager
    def transaction(self):
        yield self.session


def test_completed_deletion_physically_removes_model_row():
    model = SimpleNamespace(
        model_ref="mdl_0123456789abcdef0123456789abcdef",
        owner_subject="inventory",
        label="daily",
        generation=3,
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        lifecycle_state="DELETING",
        deletion_requested_at=datetime(2026, 8, 19, tzinfo=UTC),
    )
    session = SessionDouble(model)
    store = PublishedModelStore(DatabaseDouble(session))

    assert store.complete_deletion(
        "mdl_0123456789abcdef0123456789abcdef"
    ) is True
    assert len(session.added) == 1
    archived = session.added[0]
    assert isinstance(archived, DeletedModel)
    assert archived.model_ref == model.model_ref
    assert archived.owner_subject == model.owner_subject
    assert archived.label == model.label
    assert archived.generation == model.generation
    assert archived.created_at == model.created_at
    assert archived.deletion_requested_at == model.deletion_requested_at
    assert archived.deleted_at >= model.deletion_requested_at
    assert session.deleted == [model]
    assert session.flushed is True


def test_repeated_completion_after_model_row_was_removed_is_a_noop():
    session = SessionDouble(None)
    store = PublishedModelStore(DatabaseDouble(session))

    assert store.complete_deletion(
        "mdl_0123456789abcdef0123456789abcdef"
    ) is False
    assert session.added == []
    assert session.deleted == []
    assert session.flushed is False
