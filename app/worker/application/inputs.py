from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO

from app.contracts.worker.v3 import WorkerContractError, parse_control_message


class DurableInputStream:
    """Consume the immutable startup snapshot and ordered control channel."""

    def __init__(self, manifest: dict, stream: BinaryIO, emitter):
        self.manifest = manifest
        self.stream = stream
        self.emitter = emitter
        self.job_id = manifest["jobId"]
        self.attempt = manifest["attempt"]
        self.attempt_id = manifest["attemptId"]
        self.input_revision = manifest["inputRevision"]
        self.closed = manifest["inputClosed"]
        self.manifest_sha256 = manifest["manifestSha256"]
        self._expected_sequence = 1
        self._inputs = list(manifest["inputs"])
        self._validate_snapshot()

    @property
    def next_ordinal(self) -> int:
        return len(self._inputs)

    @property
    def inputs(self) -> tuple[dict, ...]:
        return tuple(self._inputs)

    def items(self) -> Iterator[dict]:
        for item in tuple(self._inputs):
            yield item
        while not self.closed:
            self.emitter.input_waiting(
                next_ordinal=self.next_ordinal,
                input_revision=self.input_revision,
            )
            line = self.stream.readline()
            if not line:
                raise WorkerContractError(
                    "worker control channel closed before input.closed"
                )
            message = parse_control_message(line)
            self._validate_envelope(message)
            if message["type"] == "input.committed":
                item = message["payload"]["input"]
                revision = message["payload"]["inputRevision"]
                ordinal = item["ordinal"]
                if ordinal < self.next_ordinal:
                    if item != self._inputs[ordinal]:
                        raise WorkerContractError(
                            "duplicate input control differs from accepted input"
                        )
                    self.input_revision = max(self.input_revision, revision)
                    self.emitter.input_ack(
                        ordinal=ordinal,
                        next_ordinal=self.next_ordinal,
                        input_revision=self.input_revision,
                    )
                    continue
                if ordinal != self.next_ordinal:
                    raise WorkerContractError(
                        "input control ordinal is not contiguous"
                    )
                self._validate_input(item)
                if revision < item["commitRevision"]:
                    raise WorkerContractError(
                        "input control revision precedes its receipt"
                    )
                self.input_revision = max(self.input_revision, revision)
                self._inputs.append(item)
                self.emitter.input_ack(
                    ordinal=ordinal,
                    next_ordinal=self.next_ordinal,
                    input_revision=self.input_revision,
                )
                yield item
                continue
            self._close(message["payload"])

    def _validate_snapshot(self) -> None:
        if [item["ordinal"] for item in self._inputs] != list(
            range(len(self._inputs))
        ):
            raise WorkerContractError(
                "startup input snapshot is not contiguous"
            )
        for item in self._inputs:
            self._validate_input(item)
            if item["commitRevision"] > self.input_revision:
                raise WorkerContractError(
                    "startup input receipt exceeds inputRevision"
                )
        if self.closed and self.manifest_sha256 is None:
            raise WorkerContractError(
                "closed startup input has no manifestSha256"
            )

    def _validate_input(self, item: dict) -> None:
        expected = self.manifest["dataContract"]["dataContractSha256"]
        if item["dataContractSha256"] != expected:
            raise WorkerContractError(
                "input data contract differs from the job"
            )

    def _validate_envelope(self, message: dict) -> None:
        identity = (
            message["jobId"],
            message["attempt"],
            message["attemptId"],
        )
        if identity != (self.job_id, self.attempt, self.attempt_id):
            raise WorkerContractError(
                "worker control identity differs from the active attempt"
            )
        if message["sequence"] != self._expected_sequence:
            raise WorkerContractError(
                "worker control sequence is not contiguous"
            )
        self._expected_sequence += 1

    def _close(self, payload: dict) -> None:
        if payload["inputRevision"] < self.input_revision:
            raise WorkerContractError(
                "input.closed revision precedes accepted inputs"
            )
        if payload["payloadCount"] != len(self._inputs):
            raise WorkerContractError(
                "input.closed payload count differs from accepted inputs"
            )
        if payload["totalRows"] != sum(
            item["rows"] for item in self._inputs
        ):
            raise WorkerContractError(
                "input.closed row count differs from accepted inputs"
            )
        if payload["totalBytes"] != sum(
            item["artifact"]["byteCount"] for item in self._inputs
        ):
            raise WorkerContractError(
                "input.closed byte count differs from accepted inputs"
            )
        self.input_revision = payload["inputRevision"]
        self.manifest_sha256 = payload["manifestSha256"]
        self.closed = True


__all__ = ["DurableInputStream"]
