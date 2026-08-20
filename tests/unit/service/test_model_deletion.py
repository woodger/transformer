from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from app.service.adapters.outbound.postgres.models import PublishedModel
from app.service.adapters.outbound.postgres.published_models import (
    PublishedModelStore,
)


class SessionDouble:
    def __init__(self, model):
        self.model = model
        self.deleted = []
        self.flushed = False

    def get(self, model_type, model_ref, *, with_for_update):
        assert model_type is PublishedModel
        assert model_ref == "mdl_0123456789abcdef0123456789abcdef"
        assert with_for_update is True
        return self.model

    def delete(self, model):
        self.deleted.append(model)

    def flush(self):
        self.flushed = True


class DatabaseDouble:
    def __init__(self, session):
        self.session = session

    @contextmanager
    def transaction(self):
        yield self.session


def test_completed_deletion_physically_removes_model_row():
    model = SimpleNamespace(lifecycle_state="DELETING")
    session = SessionDouble(model)
    store = PublishedModelStore(DatabaseDouble(session))

    assert store.complete_deletion(
        "mdl_0123456789abcdef0123456789abcdef"
    ) is True
    assert session.deleted == [model]
    assert session.flushed is True


def test_repeated_completion_after_model_row_was_removed_is_a_noop():
    session = SessionDouble(None)
    store = PublishedModelStore(DatabaseDouble(session))

    assert store.complete_deletion(
        "mdl_0123456789abcdef0123456789abcdef"
    ) is False
    assert session.deleted == []
    assert session.flushed is False
