from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from types import MappingProxyType

import psycopg

from app.database.config import DatabaseConfig
from app.database.tokens import AccessTokenStore
from app.flight.observability import JsonLogger

_NOTIFY_CHANNEL = "transformer_auth_tokens"


@dataclass(frozen=True)
class AuthIdentity:
    token_id: str
    subject: str


class AccessTokenCache:
    """Immutable digest index swapped atomically after PostgreSQL changes."""

    def __init__(self):
        self._entries = MappingProxyType({})
        self._lock = threading.Lock()

    def reload(self, store: AccessTokenStore) -> int:
        entries = {
            _digest(token): AuthIdentity(token_id=token_id, subject=subject)
            for token, token_id, subject in store.active_credentials()
        }
        with self._lock:
            self._entries = MappingProxyType(entries)
        return len(entries)

    def lookup(self, token: str) -> AuthIdentity | None:
        entries = self._entries
        return entries.get(_digest(token))


class AccessTokenCacheService:
    """Keep the RAM cache current via PostgreSQL LISTEN/NOTIFY."""

    def __init__(
        self,
        database_config: DatabaseConfig,
        store: AccessTokenStore,
        cache: AccessTokenCache,
        *,
        logger: JsonLogger | None = None,
    ):
        self.database_config = database_config
        self.store = store
        self.cache = cache
        self.logger = logger or JsonLogger()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    def start(self, timeout: float = 10.0) -> AccessTokenCacheService:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="transformer-auth-token-listener",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.shutdown(timeout=2.0)
            if self._startup_error is not None:
                raise RuntimeError("failed to initialize API token cache") from self._startup_error
            raise RuntimeError("timed out while initializing API token cache")
        if self._startup_error is not None:
            error = self._startup_error
            self.shutdown(timeout=2.0)
            raise RuntimeError("failed to initialize API token cache") from error
        return self

    def shutdown(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("API token listener did not stop before timeout")
            self._thread = None

    def _run(self) -> None:
        initial = True
        retry_seconds = 0.25
        while not self._stop.is_set():
            try:
                with psycopg.connect(
                    **self.database_config.psycopg_parameters,
                    autocommit=True,
                ) as connection:
                    connection.execute(f"LISTEN {_NOTIFY_CHANNEL}")
                    count = self.cache.reload(self.store)
                    self.logger.event("flight.auth.cache.loaded", activeTokens=count)
                    self._startup_error = None
                    self._ready.set()
                    initial = False
                    retry_seconds = 0.25
                    while not self._stop.is_set():
                        notifications = tuple(
                            connection.notifies(timeout=1.0, stop_after=1)
                        )
                        if notifications:
                            count = self.cache.reload(self.store)
                            self.logger.event(
                                "flight.auth.cache.reloaded",
                                activeTokens=count,
                            )
            except BaseException as exc:
                self._startup_error = exc
                self.logger.event(
                    "flight.auth.cache.connection_lost",
                    errorType=type(exc).__name__,
                )
                if initial and retry_seconds >= 4.0:
                    self._ready.set()
                    return
                if self._stop.wait(retry_seconds):
                    return
                retry_seconds = min(retry_seconds * 2, 4.0)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()
