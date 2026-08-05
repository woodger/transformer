from __future__ import annotations

from logging.config import dictConfig

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.schema import CreateSchema

from app.database.config import load_database_config
from app.database.models import Base

config = context.config
dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "generic": {
                "format": "%(levelname)-5.5s [%(name)s] %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "NOTSET",
                "formatter": "generic",
                "stream": "ext://sys.stderr",
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
        "loggers": {
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": [],
                "propagate": True,
            },
            "alembic": {
                "level": "INFO",
                "handlers": [],
                "propagate": True,
            },
        },
    }
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    database_config = config.attributes.get("database_config") or load_database_config()
    schema = config.attributes.get("schema", database_config.schema)
    context.configure(
        url=database_config.url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema=schema,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    database_config = config.attributes.get("database_config") or load_database_config()
    schema = config.attributes.get("schema", database_config.schema)
    engine = create_engine(database_config.url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(CreateSchema(schema, if_not_exists=True))
            connection.commit()
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                include_schemas=True,
                version_table_schema=schema,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
