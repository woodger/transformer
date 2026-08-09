"""Compatibility imports for the PostgreSQL service adapter."""

from app.service.adapters.outbound.postgres.config import (
    DatabaseConfig,
    load_database_config,
)
from app.service.adapters.outbound.postgres.session import Database

__all__ = ["Database", "DatabaseConfig", "load_database_config"]
