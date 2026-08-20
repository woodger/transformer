from types import SimpleNamespace

import pytest

import app.admin.bootstrap.models as models_command
from app.service.domain.model import (
    ModelDeletionBlocked,
    ModelLifecycleState,
)
from app.service.domain.records import ModelLifecycleRecord


def _record(state=ModelLifecycleState.AVAILABLE):
    return ModelLifecycleRecord(
        model_ref="mdl_0123456789abcdef0123456789abcdef",
        owner_subject="inventory",
        label="daily",
        generation=3,
        state=state,
        created_at=1.0,
        deletion_requested_at=None,
        deleted_at=None,
    )


def _wire(monkeypatch, store):
    closed = []

    class DatabaseDouble:
        def __init__(self, config):
            self.config = config

        def close(self):
            closed.append(True)

    monkeypatch.setattr(models_command, "load_database_config", object)
    monkeypatch.setattr(
        models_command,
        "require_current_schema",
        lambda _config: None,
    )
    monkeypatch.setattr(models_command, "Database", DatabaseDouble)
    monkeypatch.setattr(
        models_command,
        "PublishedModelStore",
        lambda _database: store,
    )
    return closed


def test_models_list_prints_lifecycle_metadata_and_closes_database(
    monkeypatch,
    capsys,
):
    class StoreDouble:
        def list_models(self):
            return [_record()]

    closed = _wire(monkeypatch, StoreDouble())

    models_command.run(SimpleNamespace(models_action="list"))

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "MODEL REF\tOWNER\tLABEL\tGENERATION\tSTATE\tCREATED AT"
    )
    assert lines[1].startswith(
        "mdl_0123456789abcdef0123456789abcdef\tinventory\tdaily\t"
        "3\tAVAILABLE\t"
    )
    assert closed == [True]


def test_models_delete_requests_exact_generation(
    monkeypatch,
    capsys,
):
    calls = []

    class StoreDouble:
        def request_deletion(self, model_ref):
            calls.append(model_ref)
            return _record(ModelLifecycleState.DELETING)

    closed = _wire(monkeypatch, StoreDouble())
    model_ref = "mdl_0123456789abcdef0123456789abcdef"

    models_command.run(SimpleNamespace(
        models_action="delete",
        model_ref=model_ref,
    ))

    assert calls == [model_ref]
    assert capsys.readouterr().out == (
        f"Model: {model_ref}\nState: DELETING\n"
    )
    assert closed == [True]


def test_models_delete_reports_expected_block_without_traceback(
    monkeypatch,
    capsys,
):
    class StoreDouble:
        def request_deletion(self, *_args, **_kwargs):
            raise ModelDeletionBlocked("model has active predict jobs")

    closed = _wire(monkeypatch, StoreDouble())

    with pytest.raises(SystemExit) as exc:
        models_command.run(SimpleNamespace(
            models_action="delete",
            model_ref="mdl_0123456789abcdef0123456789abcdef",
        ))

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        "ERROR: model has active predict jobs\n"
    )
    assert closed == [True]
