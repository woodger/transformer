import hashlib
import json
import os
from dataclasses import dataclass
from typing import BinaryIO, Protocol, cast

import pyarrow as pa
import pyarrow.ipc as ipc

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v3 import ModelContract
from app.service.adapters.inbound.flight.arrow import ArrowStats, InputBatchValidator
from app.service.adapters.inbound.flight.configuration import FlightUploadLimits
from app.service.adapters.inbound.flight.validation import validate_upload_metadata
from app.service.application.messages.inputs import (
    CommittedInput,
    InputPayloadReceipt,
    InputUploadAuthorization,
    InputUploadMetadata,
)
from app.service.application.ports.input_uploads import InputArtifactStore
from app.service.application.services.input_upload import (
    InputKindMismatch,
    InputUploadLifecycle,
)
from app.service.domain.errors import (
    ServiceError,
    conflict,
    invalid,
    resource_exhausted,
)
from app.service.domain.job import ErrorCode


class _ArrowBuffer(Protocol):
    def to_pybytes(self) -> bytes: ...


class FlightStreamChunk(Protocol):
    app_metadata: _ArrowBuffer | None
    data: pa.RecordBatch | None


class FlightStreamReader(Protocol):
    schema: pa.Schema

    def read_chunk(self) -> FlightStreamChunk: ...


class _IpcWriter(Protocol):
    def write_batch(self, batch: pa.RecordBatch) -> None: ...

    def close(self) -> None: ...


class _Closeable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class UploadOutcome:
    record: CommittedInput
    committed_now: bool


class InputUploadSession:
    """Own the transport staging of exactly one authorized DoPut."""

    def __init__(
        self,
        config: FlightUploadLimits,
        lifecycle: InputUploadLifecycle,
        artifact_stores: dict[str, InputArtifactStore],
        *,
        owner: str,
        job_id: str,
        ordinal: int,
        reader: FlightStreamReader,
    ) -> None:
        self.config = config
        self.lifecycle = lifecycle
        self.artifact_stores = artifact_stores
        self.owner = owner
        self.job_id = job_id
        self.ordinal = ordinal
        self.reader = reader

        self.authorization: InputUploadAuthorization | None = None
        self.artifact_store: InputArtifactStore | None = None
        self.destination: str | None = None
        self.temporary_file: BinaryIO | None = None
        self.temporary_path: str | None = None
        self.ipc_writer: _IpcWriter | None = None
        self.validator: InputBatchValidator | None = None
        self.committed = False
        self.destination_published = False

    def run(self) -> UploadOutcome:
        try:
            while True:
                try:
                    chunk = self.reader.read_chunk()
                except StopIteration:
                    break

                if self.authorization is not None:
                    # Отмена может зафиксироваться, пока транспортный поток ещё
                    # передаёт фрагменты. Финальная фиксация остаётся авторитетной
                    # границей в гонке после последнего фрагмента.
                    self.lifecycle.assert_accepting(self.authorization)

                if chunk.app_metadata is not None:
                    if self.authorization is not None:
                        raise invalid(
                            "DoPut application metadata must be sent exactly once"
                        )
                    metadata = InputUploadMetadata(**validate_upload_metadata(
                        _parse_metadata(chunk.app_metadata)
                    ))
                    if metadata.job_id != self.job_id:
                        raise invalid(
                            "metadata jobId does not match descriptor"
                        )
                    if metadata.ordinal != self.ordinal:
                        raise invalid(
                            "metadata ordinal does not match descriptor"
                        )
                    try:
                        self.authorization = self.lifecycle.authorize(
                            self.owner,
                            self.job_id,
                            metadata,
                        )
                    except InputKindMismatch as exc:
                        raise invalid(
                            "metadata schemaId does not match job operation"
                        ) from exc
                    self.artifact_store = self.artifact_stores[
                        self.authorization.storage_class
                    ]
                    feature_dim = self.authorization.job.model_config.feature_dim
                    self.validator = InputBatchValidator(
                        self.authorization.job.operation,
                        self.reader.schema,
                        source_encoding=(
                            self.authorization.job.source_encoding
                        ),
                        target_contract=ModelContract.from_document(
                            self.authorization.job.model_contract
                        ).target_contract,
                        seq_len=(
                            self.authorization.job.model_config.seq_len
                        ),
                        expected_feature_dim=feature_dim,
                        max_batch_bytes=self.config.max_batch_bytes,
                        max_payload_bytes=self.config.max_payload_bytes,
                        max_rows=self.config.max_rows_per_payload,
                    )
                    self.destination = (
                        self.artifact_store.input_candidate_path(
                            self.job_id,
                            self.ordinal,
                            metadata.payload_id,
                            self.authorization.upload_token,
                        )
                    )
                    self.temporary_file, self.temporary_path = (
                        self.artifact_store.create_temporary(
                            self.destination
                        )
                    )
                    if self.authorization.existing is None:
                        self.lifecycle.reserve(
                            self.authorization,
                            self.artifact_store.relative_path(
                                self.destination
                            ),
                        )
                    self.ipc_writer = cast(
                        _IpcWriter,
                        ipc.new_file(  # pyright: ignore[reportUnknownMemberType]
                            self.temporary_file,
                            self.reader.schema,
                        ),
                    )
                    _enforce_staged_size(
                        self.temporary_file,
                        self.config.max_payload_bytes,
                    )

                if chunk.data is not None:
                    if self.authorization is None:
                        raise invalid(
                            "DoPut application metadata must precede RecordBatch data"
                        )
                    validator = self.validator
                    ipc_writer = self.ipc_writer
                    temporary_file = self.temporary_file
                    if (
                        validator is None
                        or ipc_writer is None
                        or temporary_file is None
                    ):
                        raise RuntimeError("authorized upload staging is incomplete")
                    validator.validate_batch(chunk.data)
                    ipc_writer.write_batch(chunk.data)
                    _enforce_staged_size(
                        temporary_file,
                        self.config.max_payload_bytes,
                    )

            if self.authorization is None:
                raise invalid("DoPut application metadata is required")

            authorization = self.authorization
            validator = self.validator
            ipc_writer = self.ipc_writer
            temporary_file = self.temporary_file
            temporary_path = self.temporary_path
            artifact_store = self.artifact_store
            destination = self.destination
            if (
                validator is None
                or ipc_writer is None
                or temporary_file is None
                or temporary_path is None
                or artifact_store is None
                or destination is None
            ):
                raise RuntimeError("authorized upload staging is incomplete")

            stats = validator.stats()
            metadata = authorization.metadata
            if stats.rows != metadata.rows:
                raise invalid(
                    f"metadata logicalRows {metadata.rows} does not match "
                    f"uploaded logical row count {stats.rows}"
                )
            if stats.chunks != metadata.chunks:
                raise invalid(
                    f"metadata chunks {metadata.chunks} does not match "
                    f"uploaded chunk count {stats.chunks}"
                )
            if stats.native_rows != metadata.native_rows:
                raise invalid(
                    "metadata nativeRows does not match uploaded native row counts"
                )

            ipc_writer.close()
            self.ipc_writer = None
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_file.close()
            self.temporary_file = None

            byte_count = os.path.getsize(temporary_path)
            if byte_count > self.config.max_payload_bytes:
                raise resource_exhausted(
                    f"payload size {byte_count} exceeds limit "
                    f"{self.config.max_payload_bytes}"
                )
            digest = _sha256_file(temporary_path)
            existing = authorization.existing
            if existing is not None:
                _validate_exact_duplicate(
                    existing,
                    metadata,
                    stats,
                    byte_count,
                    digest,
                )
                _validate_committed_artifact(
                    existing,
                    artifact_store.absolute_path(
                        existing.relative_path
                    ),
                )
                os.unlink(temporary_path)
                self.temporary_path = None
                return UploadOutcome(
                    self.lifecycle.replay(authorization),
                    False,
                )

            # Метод durable_create может выбросить исключение после публикации при fsync
            # родительского каталога. Сначала помечаем файл, чтобы очистка считала
            # конечное имя кандидатом, чью принадлежность журналу нужно проверить.
            self.destination_published = True
            artifact_store.durable_create(
                temporary_path,
                destination,
            )
            self.temporary_path = None
            receipt = InputPayloadReceipt(
                relative_path=artifact_store.relative_path(
                    destination
                ),
                chunks=stats.chunks,
                rows=stats.rows,
                native_rows=stats.native_rows,
                first_range_ordinal=stats.first_range_ordinal,
                first_example_offset=stats.first_example_offset,
                last_range_ordinal=stats.last_range_ordinal,
                next_example_offset=stats.next_example_offset,
                batches=stats.batches,
                byte_count=byte_count,
                sha256=digest,
                schema_fingerprint=stats.schema_fingerprint,
            )
            try:
                record = self.lifecycle.commit(
                    self.authorization,
                    receipt,
                )
            except BaseException:
                durable_exists = _committed_record_exists(
                    self.lifecycle,
                    authorization,
                )
                if durable_exists:
                    self.committed = True
                else:
                    _best_effort_remove(
                        artifact_store,
                        destination,
                    )
                raise
            self.committed = True
            return UploadOutcome(record, True)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if self.ipc_writer is not None:
            _best_effort_close(self.ipc_writer)
        if self.temporary_file is not None and not self.temporary_file.closed:
            _best_effort_close(self.temporary_file)
        if self.temporary_path is not None:
            _best_effort_unlink(self.temporary_path)
        if self.authorization is None or self.committed:
            return
        durable_exists = _committed_record_exists(
            self.lifecycle,
            self.authorization,
        )
        if not durable_exists:
            if self.destination_published:
                if self.artifact_store is None or self.destination is None:
                    raise RuntimeError("published upload destination is unavailable")
                _best_effort_remove(
                    self.artifact_store,
                    self.destination,
                )
            _best_effort_abort(
                self.lifecycle,
                self.authorization.upload_token,
            )


def _parse_metadata(buffer: _ArrowBuffer) -> JsonObject:
    payload = buffer.to_pybytes()
    if len(payload) > 64 * 1024:
        raise invalid("DoPut application metadata exceeds 65536 bytes")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise invalid(
            "DoPut application metadata must be valid UTF-8 JSON"
        ) from exc
    if not isinstance(document, dict):
        raise invalid("DoPut application metadata must be a JSON object")
    return cast(JsonObject, document)


def _validate_exact_duplicate(
    existing: CommittedInput,
    metadata: InputUploadMetadata,
    stats: ArrowStats,
    byte_count: int,
    digest: str,
) -> None:
    matches = (
        existing.payload_id == metadata.payload_id
        and existing.ordinal == metadata.ordinal
        and existing.schema_id == metadata.schema_id
        and existing.data_contract_sha256
        == metadata.data_contract_sha256
        and existing.chunks == stats.chunks
        and existing.rows == stats.rows
        and existing.native_rows == stats.native_rows
        and existing.first_range_ordinal == stats.first_range_ordinal
        and existing.first_example_offset == stats.first_example_offset
        and existing.last_range_ordinal == stats.last_range_ordinal
        and existing.next_example_offset == stats.next_example_offset
        and existing.batches == stats.batches
        and existing.byte_count == byte_count
        and existing.sha256 == digest
        and existing.schema_fingerprint == stats.schema_fingerprint
    )
    if not matches:
        raise conflict(
            "input ordinal or payloadId conflicts with committed input"
        )


def _validate_committed_artifact(
    existing: CommittedInput,
    destination: str,
) -> None:
    code = (
        ErrorCode.RECOVERY_INPUT_UNAVAILABLE
        if existing.storage_class == "recovery"
        else ErrorCode.INTERNAL
    )
    try:
        valid = (
            os.path.isfile(destination)
            and os.path.getsize(destination) == existing.byte_count
            and _sha256_file(destination) == existing.sha256
        )
    except OSError as exc:
        raise ServiceError(
            code,
            "committed input artifact is unavailable",
        ) from exc
    if not valid:
        raise ServiceError(
            code,
            "committed input artifact is unavailable",
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enforce_staged_size(file: BinaryIO, maximum: int) -> None:
    size = file.tell()
    if size > maximum:
        raise resource_exhausted(
            f"staged IPC payload size {size} exceeds limit {maximum}"
        )


def _committed_record_exists(
    lifecycle: InputUploadLifecycle,
    authorization: InputUploadAuthorization,
) -> bool:
    try:
        return lifecycle.find_committed(authorization) is not None
    except Exception:
        # Сбой поиска не доказывает отсутствия фиксации. Сохраняем финальный
        # артефакт, чтобы сверка при старте могла разрешить принадлежность.
        return True


def _best_effort_close(resource: _Closeable) -> None:
    try:
        resource.close()
    except BaseException:
        pass


def _best_effort_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except (FileNotFoundError, OSError):
        pass


def _best_effort_remove(store: InputArtifactStore, path: str) -> None:
    try:
        store.remove(path)
    except (FileNotFoundError, OSError):
        pass


def _best_effort_abort(
    lifecycle: InputUploadLifecycle,
    upload_token: str,
) -> None:
    try:
        lifecycle.abort(upload_token)
    except Exception:
        pass
