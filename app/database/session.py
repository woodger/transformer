from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.config import DatabaseConfig


class Database:
    """Own the PostgreSQL connection pool and short-lived ORM sessions."""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        engine: Engine | None = None,
    ):
        self.config = config
        self.engine = engine or create_engine(
            config.url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
        if config.schema != "transformer":
            self.engine = self.engine.execution_options(
                schema_translate_map={"transformer": config.schema}
            )
        self._sessions = sessionmaker(
            self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def session(self) -> Session:
        return self._sessions()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.session() as session:
            with session.begin():
                yield session

    def close(self) -> None:
        self.engine.dispose()
