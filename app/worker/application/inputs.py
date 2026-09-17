from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO

from app.contracts.flight.v16.source_encoding import feature_block_dimensions
from app.contracts.json_types import JsonObject
from app.contracts.worker.v14 import WorkerContractError, parse_control_message
from app.worker.application.documents import (
    boolean_field as _boolean_field,
    integer_field as _integer_field,
    integer_list as _integer_list,
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
        data_contract = _object_field(manifest, "dataContract")
        self._feature_block_count = len(feature_block_dimensions(
            _object_field(manifest, "sourceEncoding"),
            feature_dim=_integer_field(data_contract, "featureDim"),
        ))
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
            if _string_field(message, "type") == "input":
                item = _object_field(payload, "input")
                revision = _integer_field(item, "commitRevision")
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
                self.input_revision = max(self.input_revision, revision)
                self._inputs.append(item)
                self._validate_range_sequence()
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
        self._validate_range_sequence()
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
        native_rows = _integer_list(item.get("nativeRows"), "nativeRows")
        if len(native_rows) != self._feature_block_count:
            raise WorkerContractError(
                "input native row counters differ from sourceEncoding"
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
        self.input_revision = input_revision
        self.manifest_sha256 = _string_field(payload, "manifestSha256")
        self.closed = True

    def _validate_range_sequence(self) -> None:
        nonempty = [
            item
            for item in self._inputs
            if _integer_field(item, "chunks") > 0
        ]
        if not nonempty:
            return
        first = nonempty[0]
        if (
            _integer_field(first, "firstRangeOrdinal") != 0
            or _integer_field(first, "firstExampleOffset") != 0
        ):
            raise WorkerContractError(
                "input range sequence must start at range and example zero"
            )
        for previous, current in zip(nonempty, nonempty[1:], strict=False):
            previous_range = _integer_field(previous, "lastRangeOrdinal")
            current_range = _integer_field(current, "firstRangeOrdinal")
            current_offset = _integer_field(current, "firstExampleOffset")
            same_range = (
                current_range == previous_range
                and current_offset
                == _integer_field(previous, "nextExampleOffset")
            )
            next_range = (
                current_range == previous_range + 1
                and current_offset == 0
            )
            if not (same_range or next_range):
                raise WorkerContractError(
                    "input range sequence is not contiguous"
                )
__all__ = ["DurableInputStream"]
