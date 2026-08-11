from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc

from app.service.adapters.inbound.flight.contract import parse_output_descriptor
from app.service.adapters.inbound.flight.errors import failed_precondition, not_found
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.domain.job import ExecutionState


class OutputHandler:
    def __init__(self, config, ledger, spool, *, metrics=None, logger=None):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()

    def get_flight_info(self, owner, descriptor):
        job_id, ordinal = parse_output_descriptor(descriptor)
        job = self.ledger.get_job(job_id, owner_subject=owner)
        if job is None:
            raise not_found("job output not found")
        if job["execution_state"] != ExecutionState.SUCCEEDED.value:
            raise failed_precondition(
                "job outputs are available only after successful execution"
            )
        output = next(
            (item for item in self.ledger.list_outputs(job_id) if item["ordinal"] == ordinal),
            None,
        )
        if output is None:
            raise not_found("job output not found")
        path = self.spool.absolute_path(output["relative_path"])
        with pa.memory_map(path, "r") as source:
            schema = ipc.RecordBatchFileReader(source).schema

        token, expires_at = self.ledger.issue_ticket(
            job_id=job_id,
            ordinal=ordinal,
            owner_subject=owner,
            ttl_seconds=self.config.ticket_ttl_seconds,
        )
        self.metrics.add("outputTicketsIssued")
        self.logger.event(
            "flight.output.ticket_issued",
            jobId=job_id,
            ordinal=ordinal,
            rows=output["rows"],
            bytes=output["bytes"],
            expiresAt=expires_at,
        )
        expiry = pa.scalar(
            datetime.fromtimestamp(expires_at, tz=UTC),
            type=pa.timestamp("s", tz="UTC"),
        )
        endpoint = flight.FlightEndpoint(
            flight.Ticket(token),
            [],
            expiration_time=expiry,
        )
        return flight.FlightInfo(
            schema,
            descriptor,
            [endpoint],
            total_records=output["rows"],
            total_bytes=output["bytes"],
            ordered=True,
        )

    def do_get(self, context, owner, ticket):
        token = ticket.ticket if hasattr(ticket, "ticket") else ticket
        if (
            not isinstance(token, (bytes, bytearray, memoryview))
            or not 1 <= len(token) <= 256
        ):
            raise not_found("output ticket not found")
        token = bytes(token)
        output = self.ledger.resolve_ticket(token, owner_subject=owner)
        path = self.spool.absolute_path(output["relative_path"])
        source = pa.memory_map(path, "r")
        try:
            reader = ipc.RecordBatchFileReader(source)
            batches = _stream_batches(
                context,
                source,
                reader,
            )
            observed_batches = self._observe_download(output, batches)
            try:
                return flight.GeneratorStream(reader.schema, observed_batches)
            except BaseException:
                observed_batches.close()
                batches.close()
                raise
        except BaseException:
            if not source.closed:
                source.close()
            raise

    def _observe_download(self, output, batches):
        rows = 0
        batch_count = 0
        byte_count = 0
        status = "OK"
        try:
            for batch in batches:
                rows += batch.num_rows
                batch_count += 1
                byte_count += batch.nbytes
                yield batch
        except flight.FlightCancelledError:
            status = "CANCELLED"
            raise
        except GeneratorExit:
            status = "CANCELLED"
            raise
        except BaseException:
            status = "ERROR"
            raise
        finally:
            batches.close()
            self._record_download(
                output,
                status=status,
                rows=rows,
                batches=batch_count,
                byte_count=byte_count,
            )

    def _record_download(
        self,
        output,
        *,
        status: str,
        rows: int,
        batches: int,
        byte_count: int,
    ) -> None:
        self.metrics.add("outputDownloads")
        self.metrics.add("outputDownloadRows", rows)
        self.metrics.add("outputDownloadBatches", batches)
        self.metrics.add("outputDownloadBytes", byte_count)
        if status == "CANCELLED":
            self.metrics.add("outputDownloadsCancelled")
        elif status != "OK":
            self.metrics.add("outputDownloadsFailed")
        fields = {
            "status": status,
            "rows": rows,
            "batches": batches,
            "bytes": byte_count,
        }
        if output.get("job_id") is not None:
            fields["jobId"] = output["job_id"]
        if output.get("ordinal") is not None:
            fields["ordinal"] = output["ordinal"]
        self.logger.event("flight.output.download_completed", **fields)


def _stream_batches(context, source, reader):
    """Keep the published IPC file open for the complete DoGet stream.

    Retention cannot remove a job while its ticket is valid, but a ticket can
    expire while a client is still downloading.  Opening the file before the
    stream is returned and retaining the handle until generator finalization
    makes the active DoGet a stable snapshot even when maintenance unlinks the
    published name concurrently.
    """
    try:
        for index in range(reader.num_record_batches):
            if context.is_cancelled():
                raise flight.FlightCancelledError(
                    "CANCELLED: output download was cancelled"
                )
            batch = reader.get_batch(index)
            yield batch
    finally:
        source.close()
