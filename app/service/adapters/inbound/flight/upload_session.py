import hashlib
import json
import os
from dataclasses import dataclass

import pyarrow.ipc as ipc

from app.service.adapters.inbound.flight.arrow import InputBatchValidator
from app.service.adapters.inbound.flight.contract import validate_upload_metadata
from app.service.application.input_models import (
    CommittedInput,
    InputPayloadReceipt,
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


@dataclass(frozen=True)
class UploadOutcome:
    record: CommittedInput
    committed_now: bool


class InputUploadSession:
    """Own the transport staging of exactly one authorized DoPut."""

    def __init__(
        self,
        config,
        lifecycle: InputUploadLifecycle,
        artifact_stores: dict[str, InputArtifactStore],
        *,
        owner,
        job_id,
        ordinal,
        reader,
    ):
        self.config = config
        self.lifecycle = lifecycle
        self.artifact_stores = artifact_stores
        self.owner = owner
        self.job_id = job_id
        self.ordinal = ordinal
        self.reader = reader

        self.authorization = None
        self.artifact_store = None
        self.destination = None
        self.temporary_file = None
        self.temporary_path = None
        self.ipc_writer = None
        self.validator = None
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
                    # Cancellation can commit while a transport stream is
                    # still delivering chunks. The final commit remains the
                    # authoritative fence for the race after the last chunk.
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
                    self.validator = InputBatchValidator(
                        self.authorization.job.operation,
                        self.reader.schema,
                        seq_len=(
                            self.authorization.job.model_config.seq_len
                        ),
                        expected_feature_dim=(
                            self.authorization.job.model_config.feature_dim
                        ),
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
                    self.ipc_writer = ipc.new_file(
                        self.temporary_file,
                        self.reader.schema,
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
                    self.validator.validate_batch(chunk.data)
                    self.ipc_writer.write_batch(chunk.data)
                    _enforce_staged_size(
                        self.temporary_file,
                        self.config.max_payload_bytes,
                    )

            if self.authorization is None:
                raise invalid("DoPut application metadata is required")

            stats = self.validator.stats()
            metadata = self.authorization.metadata
            if stats.rows != metadata.rows:
                raise invalid(
                    f"metadata rows {metadata.rows} does not match uploaded "
                    f"row count {stats.rows}"
                )

            self.ipc_writer.close()
            self.ipc_writer = None
            self.temporary_file.flush()
            os.fsync(self.temporary_file.fileno())
            self.temporary_file.close()
            self.temporary_file = None

            byte_count = os.path.getsize(self.temporary_path)
            if byte_count > self.config.max_payload_bytes:
                raise resource_exhausted(
                    f"payload size {byte_count} exceeds limit "
                    f"{self.config.max_payload_bytes}"
                )
            digest = _sha256_file(self.temporary_path)
            feature_dim = (
                None
                if stats.src_width is None
                else stats.src_width
                // self.authorization.job.model_config.seq_len
            )
            if stats.src_width is None or feature_dim is None:
                raise invalid("input source width is unavailable")

            existing = self.authorization.existing
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
                    self.artifact_store.absolute_path(
                        existing.relative_path
                    ),
                )
                os.unlink(self.temporary_path)
                self.temporary_path = None
                return UploadOutcome(
                    self.lifecycle.replay(self.authorization),
                    False,
                )

            # durable_create can raise after publication while fsyncing the
            # parent directory. Mark it first so cleanup treats the final name
            # as a candidate whose ledger ownership must be checked.
            self.destination_published = True
            self.artifact_store.durable_create(
                self.temporary_path,
                self.destination,
            )
            self.temporary_path = None
            receipt = InputPayloadReceipt(
                relative_path=self.artifact_store.relative_path(
                    self.destination
                ),
                rows=stats.rows,
                batches=stats.batches,
                byte_count=byte_count,
                sha256=digest,
                schema_fingerprint=stats.schema_fingerprint,
                source_width=stats.src_width,
                feature_dim=feature_dim,
            )
            try:
                record = self.lifecycle.commit(
                    self.authorization,
                    receipt,
                )
            except BaseException:
                durable = _committed_record(
                    self.lifecycle,
                    self.authorization,
                )
                if durable is not None:
                    self.committed = True
                else:
                    _best_effort_remove(
                        self.artifact_store,
                        self.destination,
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
        durable = _committed_record(
            self.lifecycle,
            self.authorization,
        )
        if durable is None:
            if self.destination_published:
                _best_effort_remove(
                    self.artifact_store,
                    self.destination,
                )
            _best_effort_abort(
                self.lifecycle,
                self.authorization.upload_token,
            )


def _parse_metadata(buffer) -> dict:
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
    return document


def _validate_exact_duplicate(existing, metadata, stats, byte_count, digest):
    matches = (
        existing.payload_id == metadata.payload_id
        and existing.ordinal == metadata.ordinal
        and existing.schema_id == metadata.schema_id
        and existing.data_contract_sha256
        == metadata.data_contract_sha256
        and existing.rows == stats.rows
        and existing.batches == stats.batches
        and existing.byte_count == byte_count
        and existing.sha256 == digest
        and existing.schema_fingerprint == stats.schema_fingerprint
        and existing.source_width == stats.src_width
    )
    if not matches:
        raise conflict(
            "input ordinal or payloadId conflicts with committed input"
        )


def _validate_committed_artifact(existing, destination: str) -> None:
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


def _enforce_staged_size(file, maximum: int) -> None:
    size = file.tell()
    if size > maximum:
        raise resource_exhausted(
            f"staged IPC payload size {size} exceeds limit {maximum}"
        )


def _committed_record(lifecycle, authorization):
    try:
        return lifecycle.find_committed(authorization)
    except Exception:
        # Unknown is distinct from a confirmed missing record. The sentinel
        # preserves the final artifact for startup reconciliation.
        return {"reconciliation": "deferred"}


def _best_effort_close(resource) -> None:
    try:
        resource.close()
    except BaseException:
        pass


def _best_effort_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except (FileNotFoundError, OSError):
        pass


def _best_effort_remove(store, path: str) -> None:
    try:
        store.remove(path)
    except (FileNotFoundError, OSError):
        pass


def _best_effort_abort(lifecycle, upload_token: str) -> None:
    try:
        lifecycle.abort(upload_token)
    except Exception:
        pass
