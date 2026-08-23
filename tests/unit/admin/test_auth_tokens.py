from datetime import UTC, datetime
from types import SimpleNamespace

import app.admin.bootstrap.auth_tokens as tokens_command
from app.admin.cli.auth_tokens import print_issued, print_list, print_revoked
from app.service.domain.access import AccessTokenRecord

TOKEN_ID = "12345678-1234-4234-8234-123456789abc"
CREATED_AT = datetime(2026, 8, 23, 12, 30, tzinfo=UTC)


def _record(*, revoked=False, token=None):
    return AccessTokenRecord(
        token_id=TOKEN_ID,
        subject="inventory",
        created_at=CREATED_AT,
        revoked_at=(CREATED_AT if revoked else None),
        token=token,
    )


def test_historical_token_output_contract(capsys):
    print_issued(_record(token="a.secret"))
    print_list([_record(), _record(revoked=True)])
    print_revoked(_record(revoked=True))

    assert capsys.readouterr().out == (
        f"Token ID: {TOKEN_ID}\n"
        "Subject: inventory\n"
        "Token: a.secret\n"
        "TOKEN ID\tSUBJECT\tCREATED AT\tSTATUS\n"
        f"{TOKEN_ID}\tinventory\t{CREATED_AT.isoformat()}\tactive\n"
        f"{TOKEN_ID}\tinventory\t{CREATED_AT.isoformat()}\trevoked\n"
        f"Revoked token: {TOKEN_ID}\n"
    )


def _wire(monkeypatch, administration):
    closed = []

    class DatabaseDouble:
        def __init__(self, config):
            self.config = config

        def close(self):
            closed.append(True)

    monkeypatch.setattr(tokens_command, "load_database_config", object)
    monkeypatch.setattr(
        tokens_command,
        "require_current_schema",
        lambda _config: None,
    )
    monkeypatch.setattr(tokens_command, "Database", DatabaseDouble)
    monkeypatch.setattr(
        tokens_command,
        "AccessTokenStore",
        lambda database: database,
    )
    monkeypatch.setattr(
        tokens_command,
        "AccessTokenAdministration",
        lambda _store: administration,
    )
    return closed


def test_issue_dispatches_subject_and_closes_database(monkeypatch, capsys):
    class Administration:
        def issue(self, subject):
            assert subject == "inventory"
            return _record(token="a.secret")

    closed = _wire(monkeypatch, Administration())

    tokens_command.run(SimpleNamespace(
        tokens_action="issue",
        subject="inventory",
    ))

    assert "Token: a.secret" in capsys.readouterr().out
    assert closed == [True]


def test_list_and_revoke_dispatch_historical_actions(monkeypatch, capsys):
    calls = []

    class Administration:
        def list(self):
            calls.append("list")
            return [_record()]

        def revoke(self, token_id):
            calls.append(("revoke", token_id))
            return _record(revoked=True)

    administration = Administration()
    closed = _wire(monkeypatch, administration)

    tokens_command.run(SimpleNamespace(tokens_action="list"))
    tokens_command.run(SimpleNamespace(
        tokens_action="revoke",
        token_id=TOKEN_ID,
    ))

    output = capsys.readouterr().out
    assert "TOKEN ID\tSUBJECT\tCREATED AT\tSTATUS" in output
    assert f"Revoked token: {TOKEN_ID}" in output
    assert calls == ["list", ("revoke", TOKEN_ID)]
    assert closed == [True, True]
