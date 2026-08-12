import json
import logging
import shutil
import sys
import threading
import time
from collections import defaultdict
from typing import TextIO

from app.contracts.json_types import JsonObject, JsonValue


class JsonLogger:
    def __init__(
        self,
        name: str = "transformer.flight",
        stream: TextIO | None = None,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.StreamHandler(stream or sys.stderr)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def event(self, event: str, **fields: object) -> None:
        payload: dict[str, object] = {
            "event": event,
            "timestamp": time.time(),
            **fields,
        }
        self._logger.info(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        )


class OperationalMetrics:
    """Small in-process aggregate metrics exposed through the health action."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: defaultdict[str, float] = defaultdict(float)
        self._rpc: defaultdict[str, int] = defaultdict(int)
        self._gauges: dict[str, JsonValue] = {}

    def record_rpc(self, method: str, status: str, elapsed_seconds: float) -> None:
        with self._lock:
            self._rpc[f"{method}:{status}"] += 1
            self._counters["rpcRequests"] += 1
            self._counters["rpcLatencySecondsTotal"] += elapsed_seconds

    def record_transition(self, from_state: str, to_state: str) -> None:
        """Count a state edge without introducing a per-job metric label."""
        with self._lock:
            self._counters[f"jobTransitions.{from_state}.{to_state}"] += 1

    def add(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set(self, name: str, value: JsonValue) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self, runtime_dir: str | None = None) -> JsonObject:
        with self._lock:
            result: JsonObject = {
                "rpc": {
                    key: value for key, value in sorted(self._rpc.items())
                },
                "counters": {
                    key: value
                    for key, value in sorted(self._counters.items())
                },
                "gauges": {
                    key: value for key, value in sorted(self._gauges.items())
                },
            }
        if runtime_dir:
            usage = shutil.disk_usage(runtime_dir)
            result["disk"] = {
                "totalBytes": usage.total,
                "usedBytes": usage.used,
                "freeBytes": usage.free,
            }
        return result
