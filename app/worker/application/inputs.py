from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO

from app.contracts.json_types import JsonObject
from app.contracts.worker.v8 import WorkerContractError, parse_control_message
from app.worker.application.documents import (
    boolean_field as _boolean_field,
    integer_field as _integer_field,
    object_field as _object_field,
    object_list as _object_list,
    optional_string_field as _optional_string_field,
    string_field as _string_field,
)
from app.worker.application.events import WorkerEventEmitter


class DurableInputStream:
    """Consume the immutable startup snapshot and ordered control channel."""

    def __init__(
        self,
        manifest: JsonObject,
        stream: BinaryIO,
        emitter: WorkerEventEmitter,
    ) -> None:
        self.manifest = manifest
        self.stream = stream
        self.emitter = emitter
        self.job_id = _string_field(manifest, "jobId")
        self.attempt = _integer_field(manifest, "attempt")
        self.attempt_id = _string_field(manifest, "attemptId")
        self.input_revision = _integer_field(manifest, "inputRevision")
        self.closed = _boolean_field(manifest, "inputClosed")
        self.manifest_sha256 = _optional_string_field(
            manifest,
            "manifestSha256",
        )
        self._expected_sequence = 1
        self._inputs = _object_list(manifest.get("inputs"), "inputs")
        self._validate_snapshot()

    @property
    def next_ordinal(self) -> int:
        return len(self._inputs)

    @property
    def inputs(self) -> tuple[JsonObject, ...]:
        return tuple(self._inputs)

    def items(self) -> Iterator[JsonObject]:
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
            payload = _object_field(message, "payload")
            if _string_field(message, "type") == "input.committed":
                item = _object_field(payload, "input")
                revision = _integer_field(payload, "inputRevision")
                ordinal = _integer_field(item, "ordinal")
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
                if revision < _integer_field(item, "commitRevision"):
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
            self._close(payload)

    def _validate_snapshot(self) -> None:
        if [_integer_field(item, "ordinal") for item in self._inputs] != list(
            range(len(self._inputs))
        ):
            raise WorkerContractError(
                "startup input snapshot is not contiguous"
            )
        for item in self._inputs:
            self._validate_input(item)
            if _integer_field(item, "commitRevision") > self.input_revision:
                raise WorkerContractError(
                    "startup input receipt exceeds inputRevision"
                )
        if self.closed and self.manifest_sha256 is None:
            raise WorkerContractError(
                "closed startup input has no manifestSha256"
            )

    def _validate_input(self, item: JsonObject) -> None:
        data_contract = _object_field(self.manifest, "dataContract")
        expected = _string_field(data_contract, "dataContractSha256")
        if _string_field(item, "dataContractSha256") != expected:
            raise WorkerContractError(
                "input data contract differs from the job"
            )

    def _validate_envelope(self, message: JsonObject) -> None:
        identity = (
            _string_field(message, "jobId"),
            _integer_field(message, "attempt"),
            _string_field(message, "attemptId"),
        )
        if identity != (self.job_id, self.attempt, self.attempt_id):
            raise WorkerContractError(
                "worker control identity differs from the active attempt"
            )
        if _integer_field(message, "sequence") != self._expected_sequence:
            raise WorkerContractError(
                "worker control sequence is not contiguous"
            )
        self._expected_sequence += 1

    def _close(self, payload: JsonObject) -> None:
        input_revision = _integer_field(payload, "inputRevision")
        if input_revision < self.input_revision:
            raise WorkerContractError(
                "input.closed revision precedes accepted inputs"
            )
        if _integer_field(payload, "payloadCount") != len(self._inputs):
            raise WorkerContractError(
                "input.closed payload count differs from accepted inputs"
            )
        if _integer_field(payload, "totalRows") != sum(
            _integer_field(item, "rows") for item in self._inputs
        ):
            raise WorkerContractError(
                "input.closed row count differs from accepted inputs"
            )
        if _integer_field(payload, "totalBytes") != sum(
            _integer_field(_object_field(item, "artifact"), "byteCount")
            for item in self._inputs
        ):
            raise WorkerContractError(
                "input.closed byte count differs from accepted inputs"
            )
        self.input_revision = input_revision
        self.manifest_sha256 = _string_field(payload, "manifestSha256")
        self.closed = True
__all__ = ["DurableInputStream"]
