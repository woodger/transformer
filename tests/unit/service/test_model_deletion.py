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
from app.service.domain.model import ModelLifecycleState


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


class ListingSessionDouble:
    def __init__(self, *, available, deleting, deleted):
        self.available = available
        self.deleting = deleting
        self.deleted = deleted

    def scalars(self, statement):
        entity = statement.column_descriptions[0]["entity"]
        if entity is DeletedModel:
            return SimpleNamespace(all=lambda: self.deleted)
        assert entity is PublishedModel
        parameter_values = set(statement.compile().params.values())
        if ModelLifecycleState.AVAILABLE.value in parameter_values:
            return SimpleNamespace(all=lambda: self.available)
        if ModelLifecycleState.DELETING.value in parameter_values:
            return SimpleNamespace(all=lambda: self.deleting)
        raise AssertionError("published model listing must filter lifecycle state")


class ListingDatabaseDouble:
    def __init__(self, session):
        self.listing_session = session

    @contextmanager
    def session(self):
        yield self.listing_session


def _model(*, generation, lifecycle_state):
    return SimpleNamespace(
        model_ref=f"mdl_{generation:032x}",
        owner_subject="inventory",
        label="daily",
        generation=generation,
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        lifecycle_state=lifecycle_state,
        deletion_requested_at=(
            None
            if lifecycle_state == ModelLifecycleState.AVAILABLE.value
            else datetime(2026, 8, 20, 1, tzinfo=UTC)
        ),
        deleted_at=datetime(2026, 8, 20, 2, tzinfo=UTC),
    )


def test_model_lists_separate_available_and_deletion_lifecycle():
    available = _model(
        generation=1,
        lifecycle_state=ModelLifecycleState.AVAILABLE.value,
    )
    deleting = _model(
        generation=2,
        lifecycle_state=ModelLifecycleState.DELETING.value,
    )
    deleted = _model(
        generation=3,
        lifecycle_state=ModelLifecycleState.DELETED.value,
    )
    session = ListingSessionDouble(
        available=[available],
        deleting=[deleting],
        deleted=[deleted],
    )
    store = PublishedModelStore(ListingDatabaseDouble(session))

    current_records = store.list_models()
    deletion_records = store.list_models(deleted=True)

    assert [record.state for record in current_records] == [
        ModelLifecycleState.AVAILABLE
    ]
    assert [record.state for record in deletion_records] == [
        ModelLifecycleState.DELETING,
        ModelLifecycleState.DELETED,
    ]
    assert deletion_records[0].deleted_at is None
    assert deletion_records[1].deleted_at is not None


def test_deleted_list_prefers_archive_during_completion_race():
    deleting = _model(
        generation=2,
        lifecycle_state=ModelLifecycleState.DELETING.value,
    )
    deleted = _model(
        generation=2,
        lifecycle_state=ModelLifecycleState.DELETED.value,
    )
    session = ListingSessionDouble(
        available=[],
        deleting=[deleting],
        deleted=[deleted],
    )
    store = PublishedModelStore(ListingDatabaseDouble(session))

    records = store.list_models(deleted=True)

    assert len(records) == 1
    assert records[0].state == ModelLifecycleState.DELETED
    assert records[0].deleted_at is not None


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
