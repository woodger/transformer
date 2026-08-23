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


def test_token_output_contract_is_aligned(capsys):
    print_issued(_record(token="a.secret"))
    print_list([_record(), _record(revoked=True)])
    print_revoked(_record(revoked=True))

    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        f"Token ID: {TOKEN_ID}",
        "Token: a.secret",
    ]
    assert lines[-1] == f"Revoked token: {TOKEN_ID}"
    for heading, active_value, revoked_value in (
        ("TOKEN ID", TOKEN_ID, TOKEN_ID),
        ("CREATED AT", CREATED_AT.isoformat(), CREATED_AT.isoformat()),
        ("STATUS", "active", "revoked"),
    ):
        offset = lines[2].index(heading)
        assert lines[3].index(active_value) == offset
        assert lines[4].index(revoked_value) == offset
    assert all("inventory" not in line for line in lines)


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
            return _record(revoked=True)

    administration = Administration()
    closed = _wire(monkeypatch, administration)

    tokens_command.run(SimpleNamespace(tokens_action="list"))
    tokens_command.run(SimpleNamespace(
        tokens_action="revoke",
        token_id=TOKEN_ID,
    ))

    output = capsys.readouterr().out
    assert "TOKEN ID" in output
    assert "CREATED AT" in output
    assert "STATUS" in output
    assert "SUBJECT" not in output
    assert "inventory" not in output
    assert f"Revoked token: {TOKEN_ID}" in output
    assert calls == ["list", ("revoke", TOKEN_ID)]
    assert closed == [True, True]
