from __future__ import annotations

from typing import Protocol

from app.service.domain.json_types import JsonObject, JsonValue


class EventLogger(Protocol):
    def event(self, event: str, **fields: object) -> None: ...


class OperationalMetricSink(Protocol):
    def record_transition(self, from_state: str, to_state: str) -> None: ...

    def add(self, name: str, value: float = 1.0) -> None: ...

    def set(self, name: str, value: JsonValue) -> None: ...

    def snapshot(self) -> JsonObject: ...


__all__ = ["EventLogger", "OperationalMetricSink"]
