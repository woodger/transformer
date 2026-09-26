from collections.abc import Generator
from datetime import UTC, datetime
from typing import Protocol, cast

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc

from app.service.adapters.inbound.flight.descriptors import parse_output_descriptor
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.application.ports.output_access import OutputArtifactStore
from app.service.application.queries.outputs import OutputAccess
from app.service.domain.errors import not_found
from app.service.domain.records import OutputRecord


class FlightOutputContext(Protocol):
    def is_cancelled(self) -> bool: ...


class _TicketValue(Protocol):
    ticket: object


class _IpcSource(Protocol):
    closed: bool

    def close(self) -> None: ...


class _RecordBatchReader(Protocol):
    schema: pa.Schema
    num_record_batches: int

    def get_batch(self, index: int) -> pa.RecordBatch: ...


class _FlightFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


_FLIGHT_CANCELLED = cast(
    type[Exception],
    vars(flight)["FlightCancelledError"],
)


class OutputHandler:
    def __init__(
        self,
        access: OutputAccess,
        artifact_store: OutputArtifactStore,
        *,
        metrics: OperationalMetrics | None = None,
        logger: JsonLogger | None = None,
    ) -> None:
        self.access = access
        self.artifact_store = artifact_store
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()

    def get_flight_info(self, owner: str, descriptor: object) -> object:
        job_descriptor = parse_output_descriptor(descriptor)
        output = self.access.locate(
            owner,
            job_descriptor.job_id,
            job_descriptor.ordinal,
        )
        path = self.artifact_store.absolute_path(output.relative_path)
        with pa.memory_map(path, "r") as source:
            reader = cast(_RecordBatchReader, ipc.RecordBatchFileReader(source))
            schema = reader.schema

        grant = self.access.issue(
            owner,
            job_descriptor.job_id,
            job_descriptor.ordinal,
        )
        self.metrics.add("outputTicketsIssued")
        self.logger.event(
            "flight.output.ticket_issued",
            jobId=job_descriptor.job_id,
            ordinal=job_descriptor.ordinal,
            rows=output.rows,
            bytes=output.byte_count,
            expiresAt=grant.expires_at,
        )
        expiry = pa.scalar(
            datetime.fromtimestamp(grant.expires_at, tz=UTC),
            type=pa.timestamp("s", tz="UTC"),
        )
        ticket_factory = cast(_FlightFactory, vars(flight)["Ticket"])
        endpoint_factory = cast(_FlightFactory, vars(flight)["FlightEndpoint"])
        endpoint = endpoint_factory(
            ticket_factory(grant.token),
            [],
            expiration_time=expiry,
        )
        flight_info_factory = cast(_FlightFactory, vars(flight)["FlightInfo"])
        return flight_info_factory(
            schema,
            descriptor,
            [endpoint],
            total_records=output.rows,
            total_bytes=output.byte_count,
            ordered=True,
        )

    def do_get(
        self,
        context: FlightOutputContext,
        owner: str,
        ticket: object,
    ) -> object:
        token = (
            cast(_TicketValue, ticket).ticket
            if hasattr(ticket, "ticket")
            else ticket
        )
        if not isinstance(token, (bytes, bytearray, memoryview)):
            raise not_found("output ticket not found")
        if isinstance(token, bytes):
            token_bytes = token
        elif isinstance(token, bytearray):
            token_bytes = bytes(token)
        else:
            token_bytes = token.tobytes()
        if not 1 <= len(token_bytes) <= 256:
            raise not_found("output ticket not found")
        output = self.access.resolve(owner, token_bytes)
        path = self.artifact_store.absolute_path(output.relative_path)
        source = cast(_IpcSource, pa.memory_map(path, "r"))
        try:
            reader = cast(_RecordBatchReader, ipc.RecordBatchFileReader(source))
            batches = _stream_batches(
                context,
                source,
                reader,
            )
            observed_batches = self._observe_download(output, batches)
            try:
                stream_factory = cast(
                    _FlightFactory,
                    vars(flight)["GeneratorStream"],
                )
                return stream_factory(reader.schema, observed_batches)
            except BaseException:
                observed_batches.close()
                batches.close()
                raise
        except BaseException:
            if not source.closed:
                source.close()
            raise

    def _observe_download(
        self,
        output: OutputRecord,
        batches: Generator[pa.RecordBatch],
    ) -> Generator[pa.RecordBatch]:
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
        except _FLIGHT_CANCELLED:
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
        output: OutputRecord,
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
        fields: dict[str, object] = {
            "status": status,
            "rows": rows,
            "batches": batches,
            "bytes": byte_count,
        }
        fields["jobId"] = output.job_id
        fields["ordinal"] = output.ordinal
        self.logger.event("flight.output.download_completed", **fields)


def _stream_batches(
    context: FlightOutputContext,
    source: _IpcSource,
    reader: _RecordBatchReader,
) -> Generator[pa.RecordBatch]:
    """Удерживать опубликованный IPC-файл открытым весь поток DoGet.

    Хранение не удаляет задачу, пока действителен её билет, но билет может
    истечь во время загрузки. Открытие файла до возврата потока и удержание
    дескриптора до завершения генератора делает активный DoGet стабильным
    снимком, даже если обслуживание одновременно удаляет опубликованное имя.
    """
    try:
        for index in range(reader.num_record_batches):
            if context.is_cancelled():
                raise _FLIGHT_CANCELLED(
                    "CANCELLED: output download was cancelled"
                )
            batch = reader.get_batch(index)
            yield batch
    finally:
        source.close()
