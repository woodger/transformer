import hashlib
import json
import os
import secrets
from dataclasses import dataclass

import pyarrow.ipc as ipc

from app.contracts.worker.v1.config import ModelConfig
from app.service.adapters.inbound.flight.arrow import InputBatchValidator
from app.service.adapters.inbound.flight.constants import (
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
    ErrorCode,
    JobState,
)
from app.service.adapters.inbound.flight.contract import validate_upload_metadata
from app.service.adapters.inbound.flight.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    invalid,
    not_found,
    resource_exhausted,
)


@dataclass(frozen=True)
class UploadOutcome:
    record: dict
    committed_now: bool


class InputUploadSession:
    """Own the temporary and durable state of exactly one DoPut."""

    def __init__(
        self,
        config,
        ledger,
        artifact_store,
        *,
        storage_class,
        owner,
        job,
        ordinal,
        reader,
    ):
        self.config = config
        self.ledger = ledger
        if storage_class not in ("runtime", "recovery"):
            raise ValueError(
                "storage_class must be runtime or recovery"
            )
        self.artifact_store = artifact_store
        self.storage_class = storage_class
        self.owner = owner
        self.job = job
        self.ordinal = ordinal
        self.reader = reader

        self.destination = self.artifact_store.input_path(
            job["job_id"],
            ordinal,
        )
        self.temporary_file = None
        self.temporary_path = None
        self.ipc_writer = None
        self.upload_token = None
        self.metadata = None
        self.existing = None
        self.committed = False
        self.destination_published = False

    def run(self) -> UploadOutcome:
        model_config = ModelConfig.from_dict(self.job["model_config"])
        validator = InputBatchValidator(
            self.job["operation"],
            self.reader.schema,
            seq_len=model_config.seq_len,
            expected_feature_dim=model_config.feature_dim,
            max_batch_bytes=self.config.max_batch_bytes,
            max_payload_bytes=self.config.max_payload_bytes,
            max_rows=self.config.max_rows_per_payload,
        )
        try:
            while True:
                try:
                    chunk = self.reader.read_chunk()
                except StopIteration:
                    break

                # Cancellation is a concurrent control-plane transaction. Do
                # not continue validating or staging transport chunks after it
                # commits; the final commit check remains the authoritative
                # guard for a race after the last chunk.
                current = self.ledger.get_job(
                    self.job["job_id"],
                    owner_subject=self.owner,
                )
                if current is None:
                    raise not_found("job not found")
                if current["state"] == JobState.CANCELLED.value:
                    raise ServiceError(
                        ErrorCode.CANCELLED,
                        "job upload was cancelled",
                    )
                if current["state"] != JobState.UPLOADING.value:
                    raise failed_precondition("job no longer accepts inputs")

                if chunk.app_metadata is not None:
                    if self.metadata is not None:
                        raise invalid(
                            "DoPut application metadata must be sent exactly once"
                        )
                    self.metadata = validate_upload_metadata(
                        _parse_metadata(chunk.app_metadata)
                    )
                    _match_upload(self.job, self.metadata, self.ordinal)
                    self.existing = _find_existing(
                        self.ledger,
                        self.job["job_id"],
                        self.ordinal,
                        self.metadata["payload_id"],
                        self.storage_class,
                    )

                    self.temporary_file, self.temporary_path = (
                        self.artifact_store.create_temporary(
                            self.destination
                        )
                    )
                    if self.existing is None:
                        self.upload_token = secrets.token_hex(24)
                        self.ledger.reserve_input(
                            job_id=self.job["job_id"],
                            payload_id=self.metadata["payload_id"],
                            ordinal=self.ordinal,
                            upload_token=self.upload_token,
                            temporary_path=self.artifact_store.relative_path(
                                self.temporary_path
                            ),
                            storage_class=self.storage_class,
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
                    if self.metadata is None:
                        raise invalid(
                            "DoPut application metadata must precede RecordBatch data"
                        )
                    validator.validate_batch(chunk.data)
                    self.ipc_writer.write_batch(chunk.data)
                    _enforce_staged_size(
                        self.temporary_file,
                        self.config.max_payload_bytes,
                    )

            if self.metadata is None:
                raise invalid("DoPut application metadata is required")

            stats = validator.stats()
            if stats.rows != self.metadata["rows"]:
                raise invalid(
                    f"metadata rows {self.metadata['rows']} does not match uploaded "
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
                else stats.src_width // model_config.seq_len
            )

            if self.existing is not None:
                _validate_exact_duplicate(
                    self.existing,
                    self.metadata,
                    stats,
                    byte_count,
                    digest,
                )
                _validate_committed_artifact(
                    self.existing,
                    self.destination,
                    self.storage_class,
                )
                os.unlink(self.temporary_path)
                self.temporary_path = None
                return UploadOutcome(self.existing, False)

            # durable_replace may raise after os.replace while fsyncing the
            # parent directory. Mark publication before the call so the
            # failure path removes any final-named, uncommitted artifact.
            self.destination_published = True
            self.artifact_store.durable_replace(
                self.temporary_path,
                self.destination,
            )
            self.temporary_path = None
            try:
                record = self.ledger.commit_input(
                    upload_token=self.upload_token,
                    relative_path=self.artifact_store.relative_path(
                        self.destination
                    ),
                    schema_id=self.metadata["schema_id"],
                    rows=stats.rows,
                    batches=stats.batches,
                    byte_count=byte_count,
                    sha256=digest,
                    schema_fingerprint=stats.schema_fingerprint,
                    source_width=stats.src_width,
                    feature_dim=feature_dim,
                    max_payloads=self.config.max_payloads_per_job,
                    max_job_bytes=self.config.max_job_bytes,
                    storage_class=self.storage_class,
                )
            except BaseException:
                # A failure can happen before the transaction commits, or
                # after commit while returning to this session. Never delete a
                # file that the ledger may already reference.
                durable = _committed_record(
                    self.ledger,
                    self.job["job_id"],
                    self.ordinal,
                    self.metadata["payload_id"],
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
        # Cleanup operations are deliberately independent. In particular,
        # ENOSPC from IPC writer finalization must not strand a reservation and
        # permanently block seal until restart.
        if self.ipc_writer is not None:
            _best_effort_close(self.ipc_writer)
        if self.temporary_file is not None and not self.temporary_file.closed:
            _best_effort_close(self.temporary_file)
        if self.temporary_path is not None:
            _best_effort_unlink(self.temporary_path)
        if self.upload_token is not None and not self.committed:
            # If commit raised after PostgreSQL committed, preserve the final
            # file and record. If the reconciliation query itself cannot run,
            # preserving the file is safer than corrupting a possible
            # committed input; startup reconciliation will decide later.
            durable = _committed_record(
                self.ledger,
                self.job["job_id"],
                self.ordinal,
                (
                    self.metadata["payload_id"]
                    if self.metadata is not None
                    else None
                ),
            )
            if durable is None:
                if self.destination_published:
                    _best_effort_remove(
                        self.artifact_store,
                        self.destination,
                    )
                _best_effort_abort(self.ledger, self.upload_token)


def _parse_metadata(buffer) -> dict:
    payload = buffer.to_pybytes()
    if len(payload) > 64 * 1024:
        raise invalid("DoPut application metadata exceeds 65536 bytes")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise invalid("DoPut application metadata must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise invalid("DoPut application metadata must be a JSON object")
    return document


def _match_upload(job: dict, metadata: dict, descriptor_ordinal: int) -> None:
    if metadata["job_id"] != job["job_id"]:
        raise invalid("metadata jobId does not match descriptor")
    if metadata["ordinal"] != descriptor_ordinal:
        raise invalid("metadata ordinal does not match descriptor")
    expected = FIT_SCHEMA_ID if job["operation"] == "fit" else PREDICT_SCHEMA_ID
    if metadata["schema_id"] != expected:
        raise invalid("metadata schemaId does not match job operation")


def _find_existing(
    ledger,
    job_id: str,
    ordinal: int,
    payload_id: str,
    storage_class: str,
):
    by_ordinal = ledger.find_input(job_id, ordinal=ordinal)
    by_payload = ledger.find_input(job_id, payload_id=payload_id)
    if by_ordinal is None and by_payload is None:
        return None
    if by_ordinal is None or by_payload is None:
        raise conflict("input ordinal or payloadId is already committed")
    if by_ordinal["ordinal"] != by_payload["ordinal"]:
        raise conflict("input ordinal and payloadId refer to different inputs")
    if (
        by_ordinal.get("storage_class") != storage_class
        or by_payload.get("storage_class") != storage_class
    ):
        raise conflict("input is committed in a different storage class")
    return by_ordinal


def _validate_exact_duplicate(existing, metadata, stats, byte_count, digest):
    matches = (
        existing["payload_id"] == metadata["payload_id"]
        and existing["ordinal"] == metadata["ordinal"]
        and existing["schema_id"] == metadata["schema_id"]
        and existing["rows"] == stats.rows
        and existing["batches"] == stats.batches
        and existing["bytes"] == byte_count
        and existing["sha256"] == digest
        and existing["schema_fingerprint"] == stats.schema_fingerprint
        and existing["source_width"] == stats.src_width
    )
    if not matches:
        raise conflict("input ordinal or payloadId conflicts with committed input")


def _validate_committed_artifact(
    existing,
    destination: str,
    storage_class: str,
) -> None:
    code = (
        ErrorCode.RECOVERY_INPUT_UNAVAILABLE
        if storage_class == "recovery"
        else ErrorCode.INTERNAL
    )
    try:
        valid = (
            os.path.isfile(destination)
            and os.path.getsize(destination) == existing["bytes"]
            and _sha256_file(destination) == existing["sha256"]
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


def _committed_record(ledger, job_id: str, ordinal: int, payload_id: str | None):
    if payload_id is None:
        return None
    try:
        by_ordinal = ledger.find_input(job_id, ordinal=ordinal)
        by_payload = ledger.find_input(job_id, payload_id=payload_id)
    except Exception:
        # Unknown is intentionally distinct from a confirmed missing record.
        # Returning a sentinel preserves the final artifact until startup
        # reconciliation can consult a healthy ledger.
        return {"reconciliation": "deferred"}
    if by_ordinal is None or by_payload is None:
        return None
    if by_ordinal["payload_id"] != payload_id or by_payload["ordinal"] != ordinal:
        return None
    return by_ordinal


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


def _best_effort_remove(spool, path: str) -> None:
    try:
        spool.remove(path)
    except (FileNotFoundError, OSError):
        pass


def _best_effort_abort(ledger, upload_token: str) -> None:
    try:
        ledger.abort_input(upload_token)
    except Exception:
        pass
