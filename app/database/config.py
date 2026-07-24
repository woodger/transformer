from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import URL

from app.config import PROJECT_ROOT

_REQUIRED_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    database: str
    user: str
    password: str = field(repr=False)
    port: int = 5432
    schema: str = "transformer"

    def __post_init__(self) -> None:
        for name in ("host", "database", "user", "password", "schema"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"PostgreSQL {name} must not be empty")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise ValueError("POSTGRES_PORT must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("POSTGRES_PORT must be between 1 and 65535")
        if not self.schema.replace("_", "a").isalnum() or not self.schema[0].isalpha():
            raise ValueError("PostgreSQL schema must be a simple SQL identifier")

    @property
    def url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    @property
    def psycopg_parameters(self) -> dict[str, str | int]:
        return {
            "host": self.host,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
            "port": self.port,
        }


def load_database_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | os.PathLike[str] | None = None,
    schema: str = "transformer",
) -> DatabaseConfig:
    """Load PostgreSQL settings without mutating ``os.environ``.

    Values already present in the process environment take precedence over
    the project-local ``.env`` file.
    """

    environment = os.environ if environ is None else environ
    if env_file is None:
        env_file = Path(PROJECT_ROOT) / ".env"
    file_values = {
        key: value
        for key, value in dotenv_values(env_file).items()
        if isinstance(value, str)
    }
    values = {**file_values, **dict(environment)}
    missing = [key for key in _REQUIRED_KEYS if not values.get(key)]
    if missing:
        raise ValueError(
            "missing PostgreSQL configuration: " + ", ".join(missing)
        )
    raw_port = values.get("POSTGRES_PORT", "5432")
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError("POSTGRES_PORT must be an integer") from exc
    return DatabaseConfig(
        host=values["POSTGRES_HOST"],
        database=values["POSTGRES_DB"],
        user=values["POSTGRES_USER"],
        password=values["POSTGRES_PASSWORD"],
        port=port,
        schema=schema,
    )
