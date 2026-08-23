from datetime import UTC, datetime
from types import SimpleNamespace

import app.admin.bootstrap.auth_tokens as tokens_command
from app.admin.cli.auth_tokens import print_issued, print_list, print_revoked
from app.service.domain.access import AccessTokenRecord

TOKEN_ID = "12345678-1234-4234-8234-123456789abc"
CREATED_AT = datetime(2026, 8, 23, 12, 30, tzinfo=UTC)
EXPIRES_AT = datetime(2026, 11, 23, 12, 30, tzinfo=UTC)


def _record(*, expired=False, token=None):
    return AccessTokenRecord(
        token_id=TOKEN_ID,
        subject="inventory",
        created_at=CREATED_AT,
        expires_at=(CREATED_AT if expired else EXPIRES_AT),
        token=token,
    )


def test_token_output_contract_is_aligned(capsys):
    print_issued(_record(token="a.secret"))
    print_list([_record()], now=CREATED_AT)
    print_revoked(_record())

    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        f"Token ID: {TOKEN_ID}",
        f"Expires: {EXPIRES_AT.isoformat()}",
        "Token: a.secret",
    ]
    assert lines[-1] == f"Revoked token: {TOKEN_ID}"
    for heading, value in (
        ("ID", TOKEN_ID),
        ("Status", "Active"),
        ("Created", CREATED_AT.isoformat()),
        ("Expires", EXPIRES_AT.isoformat()),
    ):
        offset = lines[3].index(heading)
        assert lines[4].index(value) == offset
    assert all("inventory" not in line for line in lines)
    assert all("Revoked" not in line for line in lines[:-1])


def test_token_list_reports_expiration_at_the_exact_boundary(capsys):
    print_list([_record(expired=True)], now=CREATED_AT)

    assert "Expired" in capsys.readouterr().out.splitlines()[1]


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


def test_issue_and_close_database(monkeypatch, capsys):
    class Administration:
        def issue(self):
            return _record(token="a.secret")

    closed = _wire(monkeypatch, Administration())

    tokens_command.run(SimpleNamespace(tokens_action="issue"))

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
            return _record()

    administration = Administration()
    closed = _wire(monkeypatch, administration)

    tokens_command.run(SimpleNamespace(tokens_action="list"))
    tokens_command.run(SimpleNamespace(
        tokens_action="revoke",
        token_id=TOKEN_ID,
    ))

    output = capsys.readouterr().out
    assert "ID" in output
    assert "Status" in output
    assert "Created" in output
    assert "Expires" in output
    assert "SUBJECT" not in output
    assert "inventory" not in output
    assert f"Revoked token: {TOKEN_ID}" in output
    assert calls == ["list", ("revoke", TOKEN_ID)]
    assert closed == [True, True]
