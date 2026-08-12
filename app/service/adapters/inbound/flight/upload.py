import errno
from collections.abc import Callable
from typing import Protocol

import pyarrow as pa

from app.contracts.json_types import JsonObject
from app.service.adapters.inbound.flight.configuration import FlightUploadLimits
from app.service.adapters.inbound.flight.constants import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
)
from app.service.adapters.inbound.flight.contract import (
    encode_document,
    parse_input_descriptor,
)
from app.service.adapters.inbound.flight.upload_session import (
    FlightStreamReader,
    InputUploadSession,
)
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.application.input_models import CommittedInput
from app.service.application.ports.input_uploads import InputArtifactStore
from app.service.application.services.input_upload import InputUploadLifecycle
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode


class PutMetadataWriter(Protocol):
    def write(self, value: object) -> object: ...


class UploadHandler:
    def __init__(
        self,
        config: FlightUploadLimits,
        lifecycle: InputUploadLifecycle,
        artifact_stores: dict[str, InputArtifactStore],
        *,
        queue_notifier: Callable[[str], None] | None = None,
        input_notifier: Callable[[str], None] | None = None,
        metrics: OperationalMetrics | None = None,
        logger: JsonLogger | None = None,
    ) -> None:
        self.config = config
        self.lifecycle = lifecycle
        self.artifact_stores = artifact_stores
        self._queue_notifier = queue_notifier
        self._input_notifier = input_notifier
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()

    def handle(
        self,
        owner: str,
        descriptor: object,
        reader: FlightStreamReader,
        writer: PutMetadataWriter,
    ) -> None:
        try:
            return self._handle(owner, descriptor, reader, writer)
        except OSError as exc:
            if exc.errno in _DISK_FULL_ERRNOS:
                raise ServiceError(
                    ErrorCode.DISK_FULL,
                    "artifact storage is full",
                ) from exc
            raise

    def _handle(
        self,
        owner: str,
        descriptor: object,
        reader: FlightStreamReader,
        writer: PutMetadataWriter,
    ) -> None:
        job_descriptor = parse_input_descriptor(descriptor)
        self.lifecycle.validate_ordinal(job_descriptor.ordinal)
        outcome = InputUploadSession(
            self.config,
            self.lifecycle,
            self.artifact_stores,
            owner=owner,
            job_id=job_descriptor.job_id,
            ordinal=job_descriptor.ordinal,
            reader=reader,
        ).run()
        record = outcome.record
        if outcome.committed_now:
            self.metrics.add("uploadBytes", record.byte_count)
            self.metrics.add("uploadRows", record.rows)
            self.metrics.add("uploadBatches", record.batches)
            self.logger.event(
                "flight.input.committed",
                jobId=record.job_id,
                payloadId=record.payload_id,
                ordinal=record.ordinal,
                rows=record.rows,
                batches=record.batches,
                bytes=record.byte_count,
            )
            if record.queued and self._queue_notifier is not None:
                self._queue_notifier(record.job_id)
            elif (
                record.frontier_advanced
                and self._input_notifier is not None
            ):
                self._input_notifier(record.job_id)

        # The durable ledger commit deliberately precedes the sole PutResult.
        writer.write(pa.py_buffer(encode_document(_put_result(record))))


def _put_result(record: CommittedInput) -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "jobId": record.job_id,
        "payloadId": record.payload_id,
        "ordinal": record.ordinal,
        "status": "committed",
        "rows": record.rows,
        "batches": record.batches,
        "bytes": record.byte_count,
        "sha256": record.sha256,
        "schemaFingerprint": record.schema_fingerprint,
        "inputRevision": record.input_revision,
        "nextInputOrdinal": record.next_input_ordinal,
        "queued": record.queued,
    }


_DISK_FULL_ERRNOS: set[int] = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}
