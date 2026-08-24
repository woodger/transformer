from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from app import config as defaults
from app.service.adapters.outbound.postgres.tokens import AccessTokenStore
from app.service.application.ports.authentication import InvalidAccessTokenError
from app.service.domain.access import AuthIdentity
from app.service.domain.authentication import AuthenticatedPrincipal


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    identity: AuthIdentity
    deadline: float


class AccessTokenCache:
    """Authenticate token digests through a bounded PostgreSQL cache-aside."""

    def __init__(
        self,
        store: AccessTokenStore,
        *,
        ttl_seconds: float = defaults.ACCESS_TOKEN_CACHE_TTL_SECONDS,
        max_entries: int = defaults.ACCESS_TOKEN_CACHE_MAX_ENTRIES,
    ) -> None:
        ttl_value = cast(object, ttl_seconds)
        if (
            isinstance(ttl_value, bool)
            or not isinstance(ttl_value, (int, float))
            or not math.isfinite(ttl_value)
            or ttl_value <= 0
        ):
            raise ValueError("access token cache TTL must be positive and finite")
        capacity_value = cast(object, max_entries)
        if (
            isinstance(capacity_value, bool)
            or not isinstance(capacity_value, int)
            or capacity_value <= 0
        ):
            raise ValueError("access token cache capacity must be positive")
        self.store = store
        self.ttl_seconds = float(ttl_value)
        self.max_entries = capacity_value
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._inflight: dict[str, Future[AuthIdentity | None]] = {}
        self._lock = threading.Lock()

    def authenticate(
        self,
        access_token: str,
        *,
        now: datetime | None = None,
        monotonic_now: float | None = None,
    ) -> AuthenticatedPrincipal:
        current_time = datetime.now(UTC) if now is None else now
        current_monotonic = (
            time.monotonic() if monotonic_now is None else monotonic_now
        )
        digest = _digest(access_token)
        identity = self._cached_identity(
            digest,
            now=current_time,
            monotonic_now=current_monotonic,
        )
        if identity is None:
            identity = self._authenticate_singleflight(
                digest,
                now=current_time,
                monotonic_now=current_monotonic,
            )
        if identity is None or identity.expires_at <= current_time:
            raise InvalidAccessTokenError("access token is inactive")
        return AuthenticatedPrincipal(owner_subject=identity.subject)

    def _authenticate_singleflight(
        self,
        digest: str,
        *,
        now: datetime,
        monotonic_now: float,
    ) -> AuthIdentity | None:
        with self._lock:
            identity = self._cached_identity_locked(
                digest,
                now=now,
                monotonic_now=monotonic_now,
            )
            if identity is not None:
                return identity
            pending = self._inflight.get(digest)
            if pending is None:
                pending = Future[AuthIdentity | None]()
                self._inflight[digest] = pending
                is_leader = True
            else:
                is_leader = False

        if not is_leader:
            return pending.result()

        try:
            identity = self.store.use_active_credential(
                digest,
                now=now,
            )
            if identity is not None and identity.expires_at <= now:
                identity = None
            with self._lock:
                if identity is not None:
                    self._store_identity_locked(
                        digest,
                        identity,
                        now=now,
                        monotonic_now=monotonic_now,
                    )
                pending.set_result(identity)
                self._inflight.pop(digest, None)
        except BaseException as exc:
            with self._lock:
                pending.set_exception(exc)
                self._inflight.pop(digest, None)
            raise
        return identity

    def _cached_identity(
        self,
        digest: str,
        *,
        now: datetime,
        monotonic_now: float,
    ) -> AuthIdentity | None:
        with self._lock:
            return self._cached_identity_locked(
                digest,
                now=now,
                monotonic_now=monotonic_now,
            )

    def _cached_identity_locked(
        self,
        digest: str,
        *,
        now: datetime,
        monotonic_now: float,
    ) -> AuthIdentity | None:
        entry = self._entries.get(digest)
        if entry is None:
            return None
        if (
            monotonic_now >= entry.deadline
            or now >= entry.identity.expires_at
        ):
            del self._entries[digest]
            return None
        self._entries.move_to_end(digest)
        return entry.identity

    def _store_identity_locked(
        self,
        digest: str,
        identity: AuthIdentity,
        *,
        now: datetime,
        monotonic_now: float,
    ) -> None:
        remaining_seconds = (identity.expires_at - now).total_seconds()
        ttl_seconds = min(self.ttl_seconds, remaining_seconds)
        if ttl_seconds <= 0:
            return
        self._entries[digest] = _CacheEntry(
            identity=identity,
            deadline=monotonic_now + ttl_seconds,
        )
        self._entries.move_to_end(digest)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


__all__ = ["AccessTokenCache"]
