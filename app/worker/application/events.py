from __future__ import annotations

import os
from typing import BinaryIO

from app.contracts.json_types import JsonObject
from app.contracts.worker.v15 import encode_event


class WorkerEventEmitter:
    """Write one ordered, identity-bound NDJSON worker event stream."""

    def __init__(
        self,
        *,
        job_id: str,
        attempt: int,
        attempt_id: str,
        stream: BinaryIO,
    ) -> None:
        self.job_id = job_id
        self.attempt = attempt
        self.attempt_id = attempt_id
        self.stream = stream
        self._sequence = 0
        self._terminal = False

    def emit(self, event_type: str, payload: JsonObject) -> None:
        if self._terminal:
            raise RuntimeError("worker event stream is already terminal")
        self._sequence += 1
        self.stream.write(encode_event(
            job_id=self.job_id,
            attempt=self.attempt,
            attempt_id=self.attempt_id,
            sequence=self._sequence,
            event_type=event_type,
            payload=payload,
        ))
        self.stream.flush()
        if event_type in ("completed", "error"):
            self._terminal = True

    def ready(self, *, next_ordinal: int, input_revision: int) -> None:
        self.emit("ready", {
            "pid": os.getpid(),
            "nextOrdinal": next_ordinal,
            "inputRevision": input_revision,
        })

    def input_ack(
        self,
        *,
        ordinal: int,
        next_ordinal: int,
        input_revision: int,
    ) -> None:
        self.emit("input.ack", {
            "ordinal": ordinal,
            "nextOrdinal": next_ordinal,
            "inputRevision": input_revision,
        })

    def input_waiting(self, *, next_ordinal: int, input_revision: int) -> None:
        self.emit("input.waiting", {
            "nextOrdinal": next_ordinal,
            "inputRevision": input_revision,
        })

    def progress(self, progress: JsonObject) -> None:
        self.emit("progress", {"progress": progress})

    def checkpoint(self, payload: JsonObject) -> None:
        self.emit("checkpoint", payload)

    def completed(self, result_manifest: JsonObject) -> None:
        self.emit("completed", {"resultManifest": result_manifest})

    def error(self, code: str, message: str) -> None:
        self.emit("error", {
            "code": code,
            "message": message[:1024],
            "detail": None,
        })


__all__ = ["WorkerEventEmitter"]
