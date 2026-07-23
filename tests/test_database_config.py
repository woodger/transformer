from pathlib import Path

import pytest

from app.database.config import DatabaseConfig, load_database_config


def _write_env(path: Path) -> None:
    path.write_text(
        "\n".join([
            "POSTGRES_HOST=file-host",
            "POSTGRES_PORT=5432",
            "POSTGRES_DB=transformer_test_file",
            "POSTGRES_USER=file-user",
            "POSTGRES_PASSWORD=file-password",
        ]),
        encoding="utf-8",
    )


def test_process_environment_overrides_env_file(tmp_path):
    env_file = tmp_path / ".env"
    _write_env(env_file)

    config = load_database_config(
        environ={
            "POSTGRES_HOST": "environment-host",
            "POSTGRES_PORT": "6543",
            "POSTGRES_DB": "transformer_test_environment",
        },
        env_file=env_file,
    )

    assert config == DatabaseConfig(
        host="environment-host",
        port=6543,
        database="transformer_test_environment",
        user="file-user",
        password="file-password",
    )


def test_missing_database_environment_is_reported_without_reading_process_env(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match=(
            "missing PostgreSQL configuration: POSTGRES_HOST, POSTGRES_DB, "
            "POSTGRES_USER, POSTGRES_PASSWORD"
        ),
    ):
        load_database_config(
            environ={},
            env_file=tmp_path / "absent.env",
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("not-a-port", "POSTGRES_PORT must be an integer"),
        ("0", "POSTGRES_PORT must be between 1 and 65535"),
        ("65536", "POSTGRES_PORT must be between 1 and 65535"),
    ),
)
def test_database_port_must_be_a_valid_tcp_port(tmp_path, value, message):
    env_file = tmp_path / ".env"
    _write_env(env_file)

    with pytest.raises(ValueError, match=message):
        load_database_config(
            environ={"POSTGRES_PORT": value},
            env_file=env_file,
        )


@pytest.mark.parametrize(
    ("schema", "message"),
    (
        ("", "schema must not be empty"),
        ("1schema", "schema must be a simple SQL identifier"),
        ("schema-name", "schema must be a simple SQL identifier"),
        ("schema.name", "schema must be a simple SQL identifier"),
    ),
)
def test_database_schema_must_be_a_simple_identifier(schema, message):
    with pytest.raises(ValueError, match=message):
        DatabaseConfig(
            host="localhost",
            database="transformer_test",
            user="transformer",
            password="secret",
            schema=schema,
        )


def test_database_config_repr_omits_password():
    config = DatabaseConfig(
        host="localhost",
        database="transformer_test",
        user="transformer",
        password="do-not-log-this-password",
    )

    assert "do-not-log-this-password" not in repr(config)
