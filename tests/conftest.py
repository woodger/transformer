import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import DropSchema

# add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))


@pytest.fixture(scope="session")
def postgres_config():
    from app.database.config import load_database_config
    from app.database.migrations import apply_migrations

    base = load_database_config()
    if not base.database.lower().startswith("transformer_test"):
        pytest.exit(
            "PostgreSQL integration is blocked: tests require a dedicated "
            "database whose name starts with 'transformer_test'; override "
            "POSTGRES_DB for the test run",
            returncode=4,
        )
    schema = f"transformer_test_{uuid.uuid4().hex}"
    config = type(base)(
        base.host,
        base.database,
        base.user,
        base.password,
        base.port,
        schema,
    )
    cleanup_engine = create_engine(config.url)
    try:
        with cleanup_engine.connect():
            pass
    except OperationalError:
        cleanup_engine.dispose()
        pytest.exit(
            "PostgreSQL integration is blocked: the dedicated test database "
            "is unavailable",
            returncode=4,
        )
    try:
        apply_migrations(config)
        yield config
    finally:
        with cleanup_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        cleanup_engine.dispose()


@pytest.fixture
def postgres_database(postgres_config):
    from app.database.models import Base
    from app.database.session import Database

    database = Database(postgres_config)
    with database.transaction() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(delete(table))
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def postgres_ledger(postgres_database):
    from app.flight.ledger import Ledger

    return Ledger(postgres_database).initialize()
