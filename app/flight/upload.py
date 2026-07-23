import errno

import pyarrow as pa

from app.flight.constants import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    ErrorCode,
    JobState,
)
from app.flight.contract import (
    encode_document,
    parse_input_descriptor,
)
from app.flight.errors import (
    ServiceError,
    failed_precondition,
    not_found,
    resource_exhausted,
)
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.upload_session import InputUploadSession


class UploadHandler:
    def __init__(self, config, ledger, spool, *, metrics=None, logger=None):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()

    def handle(self, owner, descriptor, reader, writer):
        try:
            return self._handle(owner, descriptor, reader, writer)
        except OSError as exc:
            if exc.errno in _DISK_FULL_ERRNOS:
                raise ServiceError(
                    ErrorCode.DISK_FULL,
                    "runtime directory is full",
                ) from exc
            raise

    def _handle(self, owner, descriptor, reader, writer):
        job_id, descriptor_ordinal = parse_input_descriptor(descriptor)
        if descriptor_ordinal >= self.config.max_payloads_per_job:
            raise resource_exhausted(
                f"input ordinal {descriptor_ordinal} exceeds the per-job payload limit"
            )
        job = self.ledger.get_job(job_id, owner_subject=owner)
        if job is None:
            raise not_found("job not found")
        if job["state"] != JobState.UPLOADING.value:
            raise failed_precondition("job no longer accepts inputs")

        outcome = InputUploadSession(
            self.config,
            self.ledger,
            self.spool,
            owner=owner,
            job=job,
            ordinal=descriptor_ordinal,
            reader=reader,
        ).run()
        record = outcome.record
        if outcome.committed_now:
            self.metrics.add("uploadBytes", record["bytes"])
            self.metrics.add("uploadRows", record["rows"])
            self.metrics.add("uploadBatches", record["batches"])
            self.logger.event(
                "flight.input.committed",
                jobId=record["job_id"],
                payloadId=record["payload_id"],
                ordinal=record["ordinal"],
                rows=record["rows"],
                batches=record["batches"],
                bytes=record["bytes"],
            )

        # The durable ledger commit deliberately precedes the sole PutResult.
        writer.write(pa.py_buffer(encode_document(_put_result(record))))


def _put_result(record: dict) -> dict:
    return {
        "contract": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "jobId": record["job_id"],
        "payloadId": record["payload_id"],
        "ordinal": record["ordinal"],
        "status": "committed",
        "rows": record["rows"],
        "batches": record["batches"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "schemaFingerprint": record["schema_fingerprint"],
    }


_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}
