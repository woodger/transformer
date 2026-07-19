import errno
import hashlib
import json
import os
import secrets
import sqlite3

import pyarrow as pa
import pyarrow.ipc as ipc

from app.flight.arrow import InputBatchValidator
from app.flight.constants import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    ErrorCode,
    FIT_SCHEMA_ID,
    JobState,
    PREDICT_SCHEMA_ID,
)
from app.flight.contract import (
    encode_document,
    parse_input_descriptor,
    validate_upload_metadata,
)
from app.flight.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    invalid,
    not_found,
    resource_exhausted,
)
from app.flight.observability import JsonLogger, OperationalMetrics
from app.training.run_config import ModelConfig


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
                    "state directory is full",
                ) from exc
            raise
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
                raise ServiceError(
                    ErrorCode.DISK_FULL,
                    "state database is full",
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

        model_config = ModelConfig.from_dict(job["model_config"])
        validator = InputBatchValidator(
            job["operation"],
            reader.schema,
            seq_len=model_config.seq_len,
            expected_feature_dim=model_config.feature_dim,
            max_batch_bytes=self.config.max_batch_bytes,
            max_payload_bytes=self.config.max_payload_bytes,
            max_rows=self.config.max_rows_per_payload,
        )
        self.spool.ensure_free_space(self.config.disk_min_free_bytes)

        destination = self.spool.input_path(job_id, descriptor_ordinal)
        temporary_file = None
        temporary_path = None
        ipc_writer = None
        upload_token = None
        metadata = None
        existing = None
        committed = False
        destination_published = False
        try:
            while True:
                try:
                    chunk = reader.read_chunk()
                except StopIteration:
                    break

                # Cancellation is a concurrent control-plane transaction.  Do
                # not continue validating or staging transport chunks after it
                # commits; the final commit check remains the authoritative
                # guard for a race after the last chunk.
                current = self.ledger.get_job(job_id, owner_subject=owner)
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
                    if metadata is not None:
                        raise invalid("DoPut application metadata must be sent exactly once")
                    metadata = validate_upload_metadata(
                        _parse_metadata(chunk.app_metadata)
                    )
                    _match_upload(job, metadata, descriptor_ordinal)
                    existing = _find_existing(
                        self.ledger,
                        job_id,
                        descriptor_ordinal,
                        metadata["payload_id"],
                    )

                    temporary_file, temporary_path = self.spool.create_temporary(destination)
                    if existing is None:
                        upload_token = secrets.token_hex(24)
                        self.ledger.reserve_input(
                            job_id=job_id,
                            payload_id=metadata["payload_id"],
                            ordinal=descriptor_ordinal,
                            upload_token=upload_token,
                            temporary_path=self.spool.relative_path(temporary_path),
                        )
                    ipc_writer = ipc.new_file(temporary_file, reader.schema)
                    _enforce_staged_size(
                        temporary_file,
                        self.config.max_payload_bytes,
                    )

                if chunk.data is not None:
                    if metadata is None:
                        raise invalid(
                            "DoPut application metadata must precede RecordBatch data"
                        )
                    validator.validate_batch(chunk.data)
                    ipc_writer.write_batch(chunk.data)
                    _enforce_staged_size(
                        temporary_file,
                        self.config.max_payload_bytes,
                    )

            if metadata is None:
                raise invalid("DoPut application metadata is required")

            stats = validator.stats()
            if stats.rows != metadata["rows"]:
                raise invalid(
                    f"metadata rows {metadata['rows']} does not match uploaded "
                    f"row count {stats.rows}"
                )

            ipc_writer.close()
            ipc_writer = None
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_file.close()
            temporary_file = None

            byte_count = os.path.getsize(temporary_path)
            if byte_count > self.config.max_payload_bytes:
                raise resource_exhausted(
                    f"payload size {byte_count} exceeds limit "
                    f"{self.config.max_payload_bytes}"
                )
            self.spool.ensure_free_space(
                self.config.disk_min_free_bytes,
            )
            digest = _sha256_file(temporary_path)
            feature_dim = (
                None
                if stats.src_width is None
                else stats.src_width // model_config.seq_len
            )

            if existing is not None:
                _validate_exact_duplicate(
                    existing,
                    metadata,
                    stats,
                    byte_count,
                    digest,
                )
                os.unlink(temporary_path)
                temporary_path = None
                result = _put_result(existing)
            else:
                # durable_replace may raise after os.replace while fsyncing the
                # parent directory.  Mark publication before the call so the
                # failure path removes any final-named, uncommitted artifact.
                destination_published = True
                self.spool.durable_replace(temporary_path, destination)
                temporary_path = None
                try:
                    record = self.ledger.commit_input(
                        upload_token=upload_token,
                        relative_path=self.spool.relative_path(destination),
                        schema_id=metadata["schema_id"],
                        rows=stats.rows,
                        batches=stats.batches,
                        byte_count=byte_count,
                        sha256=digest,
                        schema_fingerprint=stats.schema_fingerprint,
                        source_width=stats.src_width,
                        feature_dim=feature_dim,
                        max_payloads=self.config.max_payloads_per_job,
                        max_job_bytes=self.config.max_job_bytes,
                    )
                except BaseException:
                    # A failure can happen before the transaction commits, or
                    # after commit while returning to this handler.  Never
                    # delete a file that the ledger may already reference.
                    durable = _committed_record(
                        self.ledger,
                        job_id,
                        descriptor_ordinal,
                        metadata["payload_id"],
                    )
                    if durable is not None:
                        committed = True
                    else:
                        _best_effort_remove(self.spool, destination)
                    raise
                committed = True
                result = _put_result(record)
                self.metrics.add("uploadBytes", byte_count)
                self.metrics.add("uploadRows", stats.rows)
                self.metrics.add("uploadBatches", stats.batches)
                self.logger.event(
                    "flight.input.committed",
                    jobId=job_id,
                    payloadId=metadata["payload_id"],
                    ordinal=descriptor_ordinal,
                    rows=stats.rows,
                    batches=stats.batches,
                    bytes=byte_count,
                )

            # The durable ledger commit deliberately precedes the sole PutResult.
            writer.write(pa.py_buffer(encode_document(result)))
        finally:
            # Cleanup operations are deliberately independent.  In
            # particular ENOSPC from IPC writer finalization must not strand a
            # reservation and permanently block seal until restart.
            if ipc_writer is not None:
                _best_effort_close(ipc_writer)
            if temporary_file is not None and not temporary_file.closed:
                _best_effort_close(temporary_file)
            if temporary_path is not None:
                _best_effort_unlink(temporary_path)
            if upload_token is not None and not committed:
                # If commit raised after SQLite committed, preserve the final
                # file and record.  If the reconciliation query itself cannot
                # run, preserving the file is safer than corrupting a possible
                # committed input; startup reconciliation will decide later.
                durable = _committed_record(
                    self.ledger,
                    job_id,
                    descriptor_ordinal,
                    metadata["payload_id"] if metadata is not None else None,
                )
                if durable is None:
                    if destination_published:
                        _best_effort_remove(self.spool, destination)
                    _best_effort_abort(self.ledger, upload_token)


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


def _find_existing(ledger, job_id: str, ordinal: int, payload_id: str):
    by_ordinal = ledger.find_input(job_id, ordinal=ordinal)
    by_payload = ledger.find_input(job_id, payload_id=payload_id)
    if by_ordinal is None and by_payload is None:
        return None
    if by_ordinal is None or by_payload is None:
        raise conflict("input ordinal or payloadId is already committed")
    if by_ordinal["ordinal"] != by_payload["ordinal"]:
        raise conflict("input ordinal and payloadId refer to different inputs")
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


_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}
