from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import secrets
import sqlite3
import time
import uuid
from typing import Callable, Iterator, Sequence

from app.flight.constants import (
    ErrorCode,
    FIT_SCHEMA_ID,
    JobState,
    PREDICT_SCHEMA_ID,
    SUPPORTED_DEVICES,
    SUPPORTED_OPERATIONS,
)
from app.flight.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
)
from app.flight.state import validate_transition


SCHEMA_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JSON_COLUMNS = {
    "model_config_json",
    "training_config_json",
    "seal_manifest_json",
    "seal_result_json",
    "start_result_json",
    "progress_json",
    "result_json",
    "metadata_json",
    "response_json",
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    owner_subject TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('fit', 'predict')),
    state TEXT NOT NULL CHECK (state IN (
        'UPLOADING', 'SEALED', 'QUEUED', 'RUNNING',
        'SUCCEEDED', 'FAILED', 'CANCELLING', 'CANCELLED'
    )),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    requested_device TEXT NOT NULL CHECK (requested_device IN ('cpu', 'cuda', 'auto')),
    selected_device TEXT CHECK (selected_device IN ('cpu', 'cuda')),
    model_label TEXT,
    input_model_ref TEXT,
    prediction_column TEXT NOT NULL,
    model_config_json TEXT,
    training_config_json TEXT,
    config_hash TEXT NOT NULL,
    source_width INTEGER CHECK (source_width IS NULL OR source_width > 0),
    feature_dim INTEGER CHECK (feature_dim IS NULL OR feature_dim > 0),
    seal_hash TEXT,
    seal_manifest_json TEXT,
    seal_result_json TEXT,
    start_result_json TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    queue_sequence INTEGER CHECK (
        queue_sequence IS NULL OR queue_sequence > 0
    ),
    error_code TEXT,
    error_message TEXT,
    result_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    sealed_at REAL,
    queued_at REAL,
    started_at REAL,
    cancel_requested_at REAL,
    finished_at REAL,
    CHECK (
        (operation = 'fit' AND model_label IS NOT NULL AND input_model_ref IS NULL)
        OR
        (operation = 'predict' AND model_label IS NULL AND input_model_ref IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS jobs_queue_idx
    ON jobs(state, selected_device, queued_at, job_id);
CREATE INDEX IF NOT EXISTS jobs_owner_state_idx
    ON jobs(owner_subject, state);

CREATE TABLE IF NOT EXISTS input_uploads (
    upload_token TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    payload_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    temporary_path TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(job_id, ordinal),
    UNIQUE(job_id, payload_id)
);

CREATE TABLE IF NOT EXISTS job_inputs (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    payload_id TEXT NOT NULL,
    schema_id TEXT NOT NULL,
    rows INTEGER NOT NULL CHECK (rows >= 0),
    batches INTEGER NOT NULL CHECK (batches >= 0),
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    sha256 TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    source_width INTEGER CHECK (source_width IS NULL OR source_width > 0),
    feature_dim INTEGER CHECK (feature_dim IS NULL OR feature_dim > 0),
    committed_at REAL NOT NULL,
    PRIMARY KEY(job_id, ordinal),
    UNIQUE(job_id, payload_id),
    UNIQUE(relative_path)
);

CREATE TABLE IF NOT EXISTS job_attempts (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    selected_device TEXT NOT NULL CHECK (selected_device IN ('cpu', 'cuda')),
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')),
    worker_id TEXT,
    pid INTEGER,
    pgid INTEGER,
    boot_id TEXT,
    process_start_ticks INTEGER CHECK (
        process_start_ticks IS NULL OR process_start_ticks > 0
    ),
    claimed_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    exit_code INTEGER,
    error_code TEXT,
    error_message TEXT,
    PRIMARY KEY(job_id, attempt)
);

CREATE TABLE IF NOT EXISTS job_outputs (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    rows INTEGER NOT NULL CHECK (rows >= 0),
    batches INTEGER NOT NULL CHECK (batches >= 0),
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    sha256 TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    published_at REAL NOT NULL,
    PRIMARY KEY(job_id, ordinal),
    UNIQUE(relative_path)
);

CREATE TABLE IF NOT EXISTS models (
    model_ref TEXT PRIMARY KEY,
    owner_subject TEXT NOT NULL,
    label TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK (generation > 0),
    checkpoint_path TEXT NOT NULL,
    metadata_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    producing_job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
    created_at REAL NOT NULL,
    UNIQUE(owner_subject, label, generation),
    UNIQUE(checkpoint_path),
    UNIQUE(metadata_path)
);

CREATE TABLE IF NOT EXISTS model_aliases (
    owner_subject TEXT NOT NULL,
    label TEXT NOT NULL,
    model_ref TEXT NOT NULL REFERENCES models(model_ref),
    updated_at REAL NOT NULL,
    PRIMARY KEY(owner_subject, label)
);

CREATE TABLE IF NOT EXISTS idempotency_records (
    owner_subject TEXT NOT NULL,
    action_name TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    job_id TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(owner_subject, action_name, idempotency_key)
);

CREATE TABLE IF NOT EXISTS output_tickets (
    ticket_hash TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    owner_subject TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(job_id, ordinal) REFERENCES job_outputs(job_id, ordinal)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS output_tickets_expiry_idx
    ON output_tickets(expires_at);
"""


class Ledger:
    """SQLite source of truth for Flight jobs and published artifacts.

    Connections are deliberately short lived.  This makes the class safe to
    call from Flight handler and worker threads without sharing sqlite3
    connection objects across threads.
    """

    def __init__(self, database_path: str, *, busy_timeout_ms: int = 5000):
        self.database_path = os.path.abspath(os.fspath(database_path))
        self.busy_timeout_ms = busy_timeout_ms

    def initialize(self) -> "Ledger":
        parent = os.path.dirname(self.database_path)
        database_existed = os.path.exists(self.database_path)
        os.makedirs(parent, exist_ok=True)
        with self.connection() as connection:
            current_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"ledger schema version {current_version} is newer than "
                    f"supported version {SCHEMA_VERSION}"
                )
            connection.executescript(_SCHEMA)
            attempt_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(job_attempts)")
            }
            if "boot_id" not in attempt_columns:
                connection.execute("ALTER TABLE job_attempts ADD COLUMN boot_id TEXT")
            if "process_start_ticks" not in attempt_columns:
                connection.execute(
                    "ALTER TABLE job_attempts ADD COLUMN process_start_ticks INTEGER "
                    "CHECK (process_start_ticks IS NULL OR process_start_ticks > 0)"
                )
            job_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
            }
            if "queue_sequence" not in job_columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN queue_sequence INTEGER "
                    "CHECK (queue_sequence IS NULL OR queue_sequence > 0)"
                )
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(queue_sequence), 0) + 1 FROM jobs"
            ).fetchone()[0]
            for row in connection.execute(
                """
                SELECT job_id FROM jobs
                WHERE state='QUEUED' AND queue_sequence IS NULL
                ORDER BY queued_at, rowid
                """
            ).fetchall():
                connection.execute(
                    "UPDATE jobs SET queue_sequence=? WHERE job_id=?",
                    (next_sequence, row["job_id"]),
                )
                next_sequence += 1
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS jobs_queue_sequence_idx "
                "ON jobs(queue_sequence) WHERE queue_sequence IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_queue_claim_idx "
                "ON jobs(state, selected_device, queue_sequence)"
            )
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()
        if not database_existed:
            _fsync_directory(parent)
        return self

    def close(self) -> None:
        # Connections are per operation and close themselves.
        return None

    def healthcheck(self) -> bool:
        with self.connection() as connection:
            connection.execute("SELECT 1 FROM jobs LIMIT 1").fetchone()
        return True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def _write(self, connection: sqlite3.Connection | None):
        if connection is not None:
            yield connection
            return
        with self.transaction() as owned:
            yield owned

    def create_job(
        self,
        *,
        job_id: str,
        owner_subject: str,
        operation: str,
        requested_device: str,
        prediction_column: str,
        config_hash: str,
        model_label: str | None = None,
        input_model_ref: str | None = None,
        model_config=None,
        training_config=None,
        now: float | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> dict:
        job_id = _canonical_uuid(job_id, "job_id")
        if not owner_subject:
            raise ValueError("owner_subject must not be empty")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        if requested_device not in SUPPORTED_DEVICES:
            raise ValueError(f"unsupported device: {requested_device}")
        _digest(config_hash, "config_hash")
        if operation == "fit" and (not model_label or input_model_ref is not None):
            raise ValueError("fit job requires model_label only")
        if operation == "predict" and (not input_model_ref or model_label is not None):
            raise ValueError("predict job requires input_model_ref only")
        timestamp = _now(now)
        try:
            with self._write(connection) as target:
                target.execute(
                    """
                    INSERT INTO jobs(
                        job_id, owner_subject, operation, state, revision,
                        requested_device, model_label, input_model_ref,
                        prediction_column, model_config_json,
                        training_config_json, config_hash, created_at, updated_at
                    ) VALUES (?, ?, ?, 'UPLOADING', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        owner_subject,
                        operation,
                        requested_device,
                        model_label,
                        input_model_ref,
                        prediction_column,
                        _json_or_none(model_config),
                        _json_or_none(training_config),
                        config_hash,
                        timestamp,
                        timestamp,
                    ),
                )
                row = target.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise conflict(f"job already exists: {job_id}") from exc
        return _decode_row(row)

    def get_job(self, job_id: str, *, owner_subject: str | None = None) -> dict | None:
        query = "SELECT * FROM jobs WHERE job_id=?"
        values: tuple = (job_id,)
        if owner_subject is not None:
            query += " AND owner_subject=?"
            values += (owner_subject,)
        with self.connection() as connection:
            return _decode_row(connection.execute(query, values).fetchone())

    def get_status_snapshot(self, job_id: str, owner_subject: str):
        """Read job, input manifest and output manifest from one SQLite snapshot."""
        with self.connection() as connection:
            connection.execute("BEGIN")
            try:
                job = connection.execute(
                    "SELECT * FROM jobs WHERE job_id=? AND owner_subject=?",
                    (job_id, owner_subject),
                ).fetchone()
                inputs = connection.execute(
                    "SELECT * FROM job_inputs WHERE job_id=? ORDER BY ordinal",
                    (job_id,),
                ).fetchall()
                outputs = connection.execute(
                    "SELECT * FROM job_outputs WHERE job_id=? ORDER BY ordinal",
                    (job_id,),
                ).fetchall()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return (
            _decode_row(job),
            [_decode_row(row) for row in inputs],
            [_decode_row(row) for row in outputs],
        )

    def list_jobs(self, states: Sequence[str | JobState] | None = None) -> list[dict]:
        query = "SELECT * FROM jobs"
        values: tuple = ()
        if states:
            state_values = tuple(JobState(value).value for value in states)
            query += f" WHERE state IN ({','.join('?' for _ in state_values)})"
            values = state_values
        query += " ORDER BY created_at, job_id"
        with self.connection() as connection:
            return [_decode_row(row) for row in connection.execute(query, values)]

    def transition_job(
        self,
        job_id: str,
        target_state: str | JobState,
        *,
        expected_revision: int | None = None,
        updates: dict | None = None,
        now: float | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> dict:
        target_state = JobState(target_state)
        timestamp = _now(now)
        with self._write(connection) as target:
            current = target.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if current is None:
                raise not_found(f"job not found: {job_id}")
            if expected_revision is not None and current["revision"] != expected_revision:
                raise failed_precondition("job revision has changed")
            validate_transition(current["state"], target_state)
            assignments = ["state=?", "revision=revision+1", "updated_at=?"]
            values: list = [target_state.value, timestamp]
            for name, value in _transition_updates(updates or {}).items():
                assignments.append(f"{name}=?")
                values.append(value)
            values.extend((job_id, current["revision"]))
            changed = target.execute(
                f"UPDATE jobs SET {', '.join(assignments)} "
                "WHERE job_id=? AND revision=?",
                values,
            ).rowcount
            if changed != 1:
                raise failed_precondition("job revision has changed")
            row = target.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode_row(row)

    def update_progress(
        self,
        job_id: str,
        progress: dict,
        *,
        now: float | None = None,
    ) -> dict:
        timestamp = _now(now)
        with self.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE jobs
                SET progress_json=?, revision=revision+1, updated_at=?
                WHERE job_id=? AND state IN ('RUNNING', 'CANCELLING')
                """,
                (_json(progress), timestamp, job_id),
            ).rowcount
            if changed != 1:
                raise failed_precondition("job is not running")
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return _decode_row(row)

    def lookup_idempotency(
        self,
        owner_subject: str,
        action_name: str,
        idempotency_key: str,
    ) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM idempotency_records
                WHERE owner_subject=? AND action_name=? AND idempotency_key=?
                """,
                (owner_subject, action_name, idempotency_key),
            ).fetchone()
        return _decode_row(row)

    def record_idempotency(
        self,
        *,
        owner_subject: str,
        action_name: str,
        idempotency_key: str,
        request_hash: str,
        response: dict,
        job_id: str | None = None,
        now: float | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[dict, bool]:
        timestamp = _now(now)
        with self._write(connection) as target:
            existing = target.execute(
                """
                SELECT * FROM idempotency_records
                WHERE owner_subject=? AND action_name=? AND idempotency_key=?
                """,
                (owner_subject, action_name, idempotency_key),
            ).fetchone()
            if existing is not None:
                decoded = _decode_row(existing)
                if decoded["request_hash"] != request_hash:
                    raise conflict("idempotency key was used for a different request")
                return decoded["response"], True
            target.execute(
                """
                INSERT INTO idempotency_records(
                    owner_subject, action_name, idempotency_key, request_hash,
                    response_json, job_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_subject,
                    action_name,
                    idempotency_key,
                    request_hash,
                    _json(response),
                    job_id,
                    timestamp,
                ),
            )
        return response, False

    def run_idempotent(
        self,
        *,
        owner_subject: str,
        action_name: str,
        idempotency_key: str,
        request_hash: str,
        mutation: Callable[[sqlite3.Connection], tuple[dict, str | None]],
        now: float | None = None,
    ) -> tuple[dict, bool]:
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM idempotency_records
                WHERE owner_subject=? AND action_name=? AND idempotency_key=?
                """,
                (owner_subject, action_name, idempotency_key),
            ).fetchone()
            if existing is not None:
                decoded = _decode_row(existing)
                if decoded["request_hash"] != request_hash:
                    raise conflict("idempotency key was used for a different request")
                return decoded["response"], True
            response, job_id = mutation(connection)
            self.record_idempotency(
                owner_subject=owner_subject,
                action_name=action_name,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                response=response,
                job_id=job_id,
                now=now,
                connection=connection,
            )
            return response, False

    def find_input(self, job_id: str, *, ordinal: int | None = None, payload_id: str | None = None):
        if ordinal is None and payload_id is None:
            raise ValueError("ordinal or payload_id is required")
        predicates = ["job_id=?"]
        values: list = [job_id]
        if ordinal is not None:
            predicates.append("ordinal=?")
            values.append(ordinal)
        if payload_id is not None:
            predicates.append("payload_id=?")
            values.append(payload_id)
        with self.connection() as connection:
            row = connection.execute(
                f"SELECT * FROM job_inputs WHERE {' AND '.join(predicates)}",
                values,
            ).fetchone()
        return _decode_row(row)

    def reserve_input(
        self,
        *,
        job_id: str,
        payload_id: str,
        ordinal: int,
        upload_token: str,
        temporary_path: str,
        now: float | None = None,
    ) -> dict:
        _nonnegative(ordinal, "ordinal")
        _canonical_uuid(payload_id, "payload_id")
        _validate_relative_path(temporary_path)
        timestamp = _now(now)
        with self.transaction() as connection:
            job = connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job["state"] != JobState.UPLOADING.value:
                raise failed_precondition("job no longer accepts inputs")
            committed = connection.execute(
                """
                SELECT 1 FROM job_inputs
                WHERE job_id=? AND (ordinal=? OR payload_id=?)
                """,
                (job_id, ordinal, payload_id),
            ).fetchone()
            if committed is not None:
                raise conflict("input ordinal or payloadId is already committed")
            try:
                connection.execute(
                    """
                    INSERT INTO input_uploads(
                        upload_token, job_id, payload_id, ordinal,
                        temporary_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_token,
                        job_id,
                        payload_id,
                        ordinal,
                        temporary_path,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise conflict("input ordinal or payloadId is being uploaded") from exc
            row = connection.execute(
                "SELECT * FROM input_uploads WHERE upload_token=?", (upload_token,)
            ).fetchone()
        return _decode_row(row)

    def abort_input(self, upload_token: str) -> str | None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT temporary_path FROM input_uploads WHERE upload_token=?",
                (upload_token,),
            ).fetchone()
            connection.execute(
                "DELETE FROM input_uploads WHERE upload_token=?", (upload_token,)
            )
        return None if row is None else row["temporary_path"]

    def commit_input(
        self,
        *,
        upload_token: str,
        relative_path: str,
        schema_id: str,
        rows: int,
        batches: int,
        byte_count: int,
        sha256: str,
        schema_fingerprint: str,
        source_width: int | None,
        feature_dim: int | None,
        max_payloads: int,
        max_job_bytes: int,
        now: float | None = None,
    ) -> dict:
        _validate_relative_path(relative_path)
        for value, name in ((rows, "rows"), (batches, "batches"), (byte_count, "bytes")):
            _nonnegative(value, name)
        _digest(sha256, "sha256")
        _digest(schema_fingerprint, "schema_fingerprint")
        timestamp = _now(now)
        with self.transaction() as connection:
            upload = connection.execute(
                "SELECT * FROM input_uploads WHERE upload_token=?", (upload_token,)
            ).fetchone()
            if upload is None:
                raise not_found("input upload reservation not found")
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (upload["job_id"],)
            ).fetchone()
            if job["state"] != JobState.UPLOADING.value:
                raise failed_precondition("job no longer accepts inputs")
            expected_schema_id = (
                FIT_SCHEMA_ID if job["operation"] == "fit" else PREDICT_SCHEMA_ID
            )
            if schema_id != expected_schema_id:
                raise failed_precondition(
                    f"schemaId {schema_id!r} does not match job operation"
                )
            totals = connection.execute(
                """
                SELECT COUNT(*) AS payloads, COALESCE(SUM(bytes), 0) AS bytes
                FROM job_inputs WHERE job_id=?
                """,
                (upload["job_id"],),
            ).fetchone()
            if totals["payloads"] + 1 > max_payloads:
                raise ServiceError(ErrorCode.RESOURCE_EXHAUSTED, "job payload quota exceeded")
            if totals["bytes"] + byte_count > max_job_bytes:
                raise ServiceError(ErrorCode.RESOURCE_EXHAUSTED, "job byte quota exceeded")
            existing_contract = connection.execute(
                """
                SELECT schema_id, schema_fingerprint FROM job_inputs
                WHERE job_id=? ORDER BY ordinal LIMIT 1
                """,
                (upload["job_id"],),
            ).fetchone()
            if existing_contract is not None and (
                existing_contract["schema_id"] != schema_id
                or existing_contract["schema_fingerprint"] != schema_fingerprint
            ):
                raise failed_precondition(
                    "input schema is inconsistent with committed inputs"
                )
            known_dimensions = connection.execute(
                """
                SELECT DISTINCT source_width, feature_dim FROM job_inputs
                WHERE job_id=? AND source_width IS NOT NULL
                """,
                (upload["job_id"],),
            ).fetchall()
            if source_width is not None and any(
                item["source_width"] != source_width
                or item["feature_dim"] != feature_dim
                for item in known_dimensions
            ):
                raise failed_precondition(
                    "input dimensions are inconsistent with committed inputs"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO job_inputs(
                        job_id, ordinal, payload_id, schema_id, rows, batches,
                        bytes, sha256, schema_fingerprint, relative_path,
                        source_width, feature_dim, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload["job_id"],
                        upload["ordinal"],
                        upload["payload_id"],
                        schema_id,
                        rows,
                        batches,
                        byte_count,
                        sha256,
                        schema_fingerprint,
                        relative_path,
                        source_width,
                        feature_dim,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise conflict("input ordinal or payloadId is already committed") from exc
            connection.execute(
                "DELETE FROM input_uploads WHERE upload_token=?", (upload_token,)
            )
            connection.execute(
                """
                UPDATE jobs SET revision=revision+1, updated_at=?
                WHERE job_id=?
                """,
                (timestamp, upload["job_id"]),
            )
            row = connection.execute(
                "SELECT * FROM job_inputs WHERE job_id=? AND ordinal=?",
                (upload["job_id"], upload["ordinal"]),
            ).fetchone()
            revision = connection.execute(
                "SELECT revision FROM jobs WHERE job_id=?", (upload["job_id"],)
            ).fetchone()["revision"]
        result = _decode_row(row)
        result["revision"] = revision
        return result

    def list_inputs(self, job_id: str) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM job_inputs WHERE job_id=? ORDER BY ordinal", (job_id,)
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def seal_job(
        self,
        job_id: str,
        *,
        manifest_hash: str,
        manifest: list[dict],
        source_width: int | None = None,
        feature_dim: int | None = None,
        result: dict | None = None,
        now: float | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[dict, bool]:
        _digest(manifest_hash, "manifest_hash")
        timestamp = _now(now)
        with self._write(connection) as target:
            job = target.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job["seal_hash"] is not None:
                if job["seal_hash"] != manifest_hash:
                    raise conflict("job was sealed with a different manifest")
                return _decode_row(job), True
            if job["state"] != JobState.UPLOADING.value:
                raise failed_precondition("job cannot be sealed from its current state")
            active = target.execute(
                "SELECT 1 FROM input_uploads WHERE job_id=? LIMIT 1", (job_id,)
            ).fetchone()
            if active is not None:
                raise failed_precondition("job has an upload in progress")
            target.execute(
                """
                UPDATE jobs SET
                    state='SEALED', revision=revision+1, updated_at=?, sealed_at=?,
                    seal_hash=?, seal_manifest_json=?, seal_result_json=?,
                    source_width=?, feature_dim=?
                WHERE job_id=? AND state='UPLOADING'
                """,
                (
                    timestamp,
                    timestamp,
                    manifest_hash,
                    _json(manifest),
                    _json_or_none(result),
                    source_width,
                    feature_dim,
                    job_id,
                ),
            )
            row = target.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode_row(row), False

    def queue_job(
        self,
        job_id: str,
        *,
        selected_device: str,
        result: dict | None = None,
        now: float | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[dict, bool]:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        timestamp = _now(now)
        with self._write(connection) as target:
            job = target.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise not_found(f"job not found: {job_id}")
            # A start that already committed remains an idempotent lifecycle
            # operation after the job advances (including later cancellation).
            # queued_at is durable evidence that SEALED -> QUEUED happened;
            # selected_device alone is not sufficient for legacy/corrupt rows.
            if job["queued_at"] is not None:
                if job["selected_device"] != selected_device:
                    raise conflict("job was started with a different device")
                return _decode_row(job), True
            if job["state"] != JobState.SEALED.value:
                raise failed_precondition("job must be SEALED before start")
            queue_sequence = target.execute(
                "SELECT COALESCE(MAX(queue_sequence), 0) + 1 FROM jobs"
            ).fetchone()[0]
            target.execute(
                """
                UPDATE jobs SET state='QUEUED', revision=revision+1,
                    selected_device=?, start_result_json=?, queued_at=?, updated_at=?,
                    queue_sequence=?
                WHERE job_id=? AND state='SEALED'
                """,
                (
                    selected_device,
                    _json_or_none(result),
                    timestamp,
                    timestamp,
                    queue_sequence,
                    job_id,
                ),
            )
            row = target.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode_row(row), False

    def claim_next_job(
        self,
        selected_device: str,
        *,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> dict | None:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        timestamp = _now(now)
        with self.transaction() as connection:
            job = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state='QUEUED' AND selected_device=?
                ORDER BY queue_sequence LIMIT 1
                """,
                (selected_device,),
            ).fetchone()
            if job is None:
                return None
            attempt = job["attempt"] + 1
            changed = connection.execute(
                """
                UPDATE jobs SET state='RUNNING', revision=revision+1,
                    attempt=?, started_at=?, updated_at=?
                WHERE job_id=? AND state='QUEUED' AND revision=?
                """,
                (attempt, timestamp, timestamp, job["job_id"], job["revision"]),
            ).rowcount
            if changed != 1:
                return None
            connection.execute(
                """
                INSERT INTO job_attempts(
                    job_id, attempt, selected_device, status, worker_id,
                    claimed_at, started_at
                ) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?)
                """,
                (
                    job["job_id"],
                    attempt,
                    selected_device,
                    worker_id,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()
        return _decode_row(row)

    def set_attempt_process(
        self,
        job_id: str,
        attempt: int,
        *,
        pid: int,
        pgid: int,
        boot_id: str,
        process_start_ticks: int,
    ) -> None:
        _positive(pid, "pid")
        _positive(pgid, "pgid")
        _positive(process_start_ticks, "process_start_ticks")
        _canonical_uuid(boot_id, "boot_id")
        with self.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE job_attempts SET pid=?, pgid=?, boot_id=?, process_start_ticks=?
                WHERE job_id=? AND attempt=? AND status='RUNNING'
                """,
                (
                    pid,
                    pgid,
                    boot_id,
                    process_start_ticks,
                    job_id,
                    attempt,
                ),
            ).rowcount
            if changed != 1:
                raise failed_precondition("job attempt is not running")

    def list_active_attempts(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT job_attempts.* FROM job_attempts
                JOIN jobs USING(job_id)
                WHERE jobs.state IN ('RUNNING', 'CANCELLING')
                    AND job_attempts.status='RUNNING'
                    AND job_attempts.attempt=jobs.attempt
                ORDER BY job_attempts.job_id, job_attempts.attempt
                """
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def finish_attempt(
        self,
        job_id: str,
        attempt: int,
        target_state: str | JobState,
        *,
        error_code: str | ErrorCode | None = None,
        error_message: str | None = None,
        exit_code: int | None = None,
        now: float | None = None,
    ) -> dict:
        """Atomically finish a failed or cancelled worker attempt."""
        target_state = JobState(target_state)
        if target_state not in (JobState.FAILED, JobState.CANCELLED):
            raise ValueError("worker attempt can finish only as FAILED or CANCELLED")
        timestamp = _now(now)
        with self.transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job["attempt"] != attempt:
                raise failed_precondition("job attempt is no longer active")
            validate_transition(job["state"], target_state)
            code = error_code.value if isinstance(error_code, ErrorCode) else error_code
            connection.execute(
                """
                UPDATE job_attempts SET status=?, finished_at=?, exit_code=?,
                    error_code=?, error_message=?
                WHERE job_id=? AND attempt=? AND status='RUNNING'
                """,
                (
                    target_state.value,
                    timestamp,
                    exit_code,
                    code,
                    error_message,
                    job_id,
                    attempt,
                ),
            )
            changed = connection.execute(
                """
                UPDATE jobs SET state=?, revision=revision+1, error_code=?,
                    error_message=?, finished_at=?, updated_at=?
                WHERE job_id=? AND revision=?
                """,
                (
                    target_state.value,
                    code,
                    error_message,
                    timestamp,
                    timestamp,
                    job_id,
                    job["revision"],
                ),
            ).rowcount
            if changed != 1:
                raise failed_precondition("job revision has changed")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode_row(row)

    def publish_outputs(
        self,
        job_id: str,
        attempt: int,
        outputs: Sequence[dict],
        *,
        result: dict,
        now: float | None = None,
    ) -> dict:
        timestamp = _now(now)
        with self.transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if (
                job["operation"] != "predict"
                or job["state"] != JobState.RUNNING.value
                or job["attempt"] != attempt
            ):
                raise failed_precondition("job is not the active running attempt")
            for output in outputs:
                _insert_output(connection, job_id, output, timestamp)
            connection.execute(
                """
                UPDATE job_attempts SET status='SUCCEEDED', finished_at=?
                WHERE job_id=? AND attempt=?
                """,
                (timestamp, job_id, attempt),
            )
            connection.execute(
                """
                UPDATE jobs SET state='SUCCEEDED', revision=revision+1,
                    result_json=?, error_code=NULL, error_message=NULL,
                    finished_at=?, updated_at=?
                WHERE job_id=? AND state='RUNNING' AND attempt=?
                """,
                (_json(result), timestamp, timestamp, job_id, attempt),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode_row(row)

    def publish_model(
        self,
        job_id: str,
        attempt: int,
        *,
        model_ref: str,
        label: str,
        generation: int | None,
        checkpoint_path: str,
        metadata_path: str,
        sha256: str,
        metadata: dict,
        result: dict,
        now: float | None = None,
    ) -> dict:
        _validate_relative_path(checkpoint_path)
        _validate_relative_path(metadata_path)
        _digest(sha256, "sha256")
        timestamp = _now(now)
        with self.transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if (
                job["operation"] != "fit"
                or job["state"] != JobState.RUNNING.value
                or job["attempt"] != attempt
            ):
                raise failed_precondition("job is not the active fit attempt")
            if label != job["model_label"]:
                raise failed_precondition("model label does not match the fit job")
            next_generation = connection.execute(
                """
                SELECT COALESCE(MAX(generation), 0) + 1 FROM models
                WHERE owner_subject=? AND label=?
                """,
                (job["owner_subject"], label),
            ).fetchone()[0]
            if generation is None:
                generation = next_generation
            elif generation != next_generation:
                raise failed_precondition(
                    f"next model generation is {next_generation}, got {generation}"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO models(
                        model_ref, owner_subject, label, generation,
                        checkpoint_path, metadata_path, sha256, metadata_json,
                        producing_job_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_ref,
                        job["owner_subject"],
                        label,
                        generation,
                        checkpoint_path,
                        metadata_path,
                        sha256,
                        _json(metadata),
                        job_id,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise conflict("model generation already exists") from exc
            connection.execute(
                """
                INSERT INTO model_aliases(owner_subject, label, model_ref, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_subject, label) DO UPDATE SET
                    model_ref=excluded.model_ref, updated_at=excluded.updated_at
                """,
                (job["owner_subject"], label, model_ref, timestamp),
            )
            connection.execute(
                """
                UPDATE job_attempts SET status='SUCCEEDED', finished_at=?
                WHERE job_id=? AND attempt=?
                """,
                (timestamp, job_id, attempt),
            )
            connection.execute(
                """
                UPDATE jobs SET state='SUCCEEDED', revision=revision+1,
                    result_json=?, finished_at=?, updated_at=?
                WHERE job_id=? AND state='RUNNING' AND attempt=?
                """,
                (_json(result), timestamp, timestamp, job_id, attempt),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode_row(row)

    def get_model(self, model_ref: str, *, owner_subject: str | None = None) -> dict | None:
        query = "SELECT * FROM models WHERE model_ref=?"
        values: tuple = (model_ref,)
        if owner_subject is not None:
            query += " AND owner_subject=?"
            values += (owner_subject,)
        with self.connection() as connection:
            return _decode_row(connection.execute(query, values).fetchone())

    def resolve_model_alias(self, owner_subject: str, label: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT models.* FROM model_aliases
                JOIN models USING(model_ref)
                WHERE model_aliases.owner_subject=? AND model_aliases.label=?
                """,
                (owner_subject, label),
            ).fetchone()
        return _decode_row(row)

    def list_outputs(self, job_id: str) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM job_outputs WHERE job_id=? ORDER BY ordinal", (job_id,)
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def issue_ticket(
        self,
        *,
        job_id: str,
        ordinal: int,
        owner_subject: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> tuple[bytes, float]:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        timestamp = _now(now)
        expires_at = timestamp + ttl_seconds
        token = secrets.token_urlsafe(32).encode("ascii")
        ticket_hash = hashlib.sha256(token).hexdigest()
        with self.transaction() as connection:
            output = connection.execute(
                """
                SELECT 1 FROM job_outputs
                JOIN jobs USING(job_id)
                WHERE job_outputs.job_id=? AND job_outputs.ordinal=?
                    AND jobs.owner_subject=? AND jobs.state='SUCCEEDED'
                """,
                (job_id, ordinal, owner_subject),
            ).fetchone()
            if output is None:
                raise not_found("published job output not found")
            connection.execute(
                """
                INSERT INTO output_tickets(
                    ticket_hash, job_id, ordinal, owner_subject,
                    expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticket_hash, job_id, ordinal, owner_subject, expires_at, timestamp),
            )
        return token, expires_at

    def resolve_ticket(
        self,
        ticket: bytes,
        *,
        owner_subject: str,
        now: float | None = None,
    ) -> dict:
        ticket_hash = hashlib.sha256(bytes(ticket)).hexdigest()
        timestamp = _now(now)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT output_tickets.owner_subject AS ticket_owner,
                    output_tickets.expires_at, job_outputs.*, jobs.state
                FROM output_tickets
                JOIN job_outputs USING(job_id, ordinal)
                JOIN jobs USING(job_id)
                WHERE ticket_hash=?
                """,
                (ticket_hash,),
            ).fetchone()
        if row is None:
            raise not_found("output ticket not found")
        if row["ticket_owner"] != owner_subject:
            raise ServiceError(ErrorCode.PERMISSION_DENIED, "output ticket belongs to another subject")
        if row["expires_at"] <= timestamp:
            raise failed_precondition("output ticket has expired")
        if row["state"] != JobState.SUCCEEDED.value:
            raise failed_precondition("job output is not available")
        return _decode_row(row)

    def delete_expired_tickets(self, *, now: float | None = None) -> int:
        with self.transaction() as connection:
            return connection.execute(
                "DELETE FROM output_tickets WHERE expires_at<=?", (_now(now),)
            ).rowcount

    def reconcile_interrupted_jobs(self, *, now: float | None = None) -> dict:
        timestamp = _now(now)
        with self.transaction() as connection:
            uploads = [
                row["temporary_path"]
                for row in connection.execute("SELECT temporary_path FROM input_uploads")
            ]
            interrupted = [
                row["job_id"]
                for row in connection.execute(
                    "SELECT job_id FROM jobs WHERE state='RUNNING' ORDER BY job_id"
                )
            ]
            cancelling = [
                row["job_id"]
                for row in connection.execute(
                    "SELECT job_id FROM jobs WHERE state='CANCELLING' ORDER BY job_id"
                )
            ]
            connection.execute(
                """
                UPDATE jobs SET state='FAILED', revision=revision+1,
                    error_code=?, error_message=?, finished_at=?, updated_at=?
                WHERE state='RUNNING'
                """,
                (
                    ErrorCode.EXECUTION_INTERRUPTED.value,
                    "worker execution was interrupted by service restart",
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE job_attempts SET status='FAILED', error_code=?,
                    error_message=?, finished_at=?
                WHERE status='RUNNING' AND job_id IN (
                    SELECT job_id FROM jobs WHERE state='FAILED'
                        AND error_code=?
                )
                """,
                (
                    ErrorCode.EXECUTION_INTERRUPTED.value,
                    "worker execution was interrupted by service restart",
                    timestamp,
                    ErrorCode.EXECUTION_INTERRUPTED.value,
                ),
            )
            connection.execute(
                """
                UPDATE jobs SET state='CANCELLED', revision=revision+1,
                    finished_at=?, updated_at=? WHERE state='CANCELLING'
                """,
                (timestamp, timestamp),
            )
            connection.execute(
                """
                UPDATE job_attempts SET status='CANCELLED', finished_at=?
                WHERE status='RUNNING' AND job_id IN (
                    SELECT job_id FROM jobs WHERE state='CANCELLED'
                )
                """,
                (timestamp,),
            )
            connection.execute("DELETE FROM input_uploads")
        return {
            "interrupted_jobs": interrupted,
            "cancelled_jobs": cancelling,
            "temporary_paths": uploads,
        }

    def referenced_paths(self) -> set[str]:
        with self.connection() as connection:
            paths = {
                row[0]
                for row in connection.execute("SELECT relative_path FROM job_inputs")
            }
            paths.update(
                row[0]
                for row in connection.execute("SELECT relative_path FROM job_outputs")
            )
            for row in connection.execute(
                "SELECT checkpoint_path, metadata_path FROM models"
            ):
                paths.update(row)
        return paths

    def delete_terminal_jobs_before(self, cutoff: float) -> list[str]:
        """Delete retained terminal jobs that do not own a model generation.

        A job remains while it has a non-expired output ticket or a recently
        created idempotency record.  Once the retention window has elapsed,
        its idempotency records are deleted in the same transaction so replay
        can never return a job ID whose row was retained away.
        """
        terminal_values = tuple(state.value for state in (
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
        ))
        with self.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT job_id FROM jobs
                WHERE state IN ({','.join('?' for _ in terminal_values)})
                    AND finished_at IS NOT NULL AND finished_at<?
                    AND NOT EXISTS (
                        SELECT 1 FROM models WHERE producing_job_id=jobs.job_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM output_tickets
                        WHERE output_tickets.job_id=jobs.job_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM idempotency_records
                        WHERE idempotency_records.job_id=jobs.job_id
                            AND idempotency_records.created_at>=?
                    )
                ORDER BY job_id
                """,
                (*terminal_values, float(cutoff), float(cutoff)),
            ).fetchall()
            job_ids = [row["job_id"] for row in rows]
            if job_ids:
                connection.execute(
                    f"DELETE FROM idempotency_records WHERE job_id IN "
                    f"({','.join('?' for _ in job_ids)})",
                    job_ids,
                )
                connection.execute(
                    f"DELETE FROM jobs WHERE job_id IN "
                    f"({','.join('?' for _ in job_ids)})",
                    job_ids,
                )
            connection.execute(
                """
                DELETE FROM idempotency_records
                WHERE job_id IS NULL AND created_at<?
                """,
                (float(cutoff),),
            )
        return job_ids


def _insert_output(connection, job_id: str, output: dict, timestamp: float) -> None:
    relative_path = output["relative_path"]
    _validate_relative_path(relative_path)
    for name in ("ordinal", "rows", "batches", "bytes"):
        _nonnegative(output[name], name)
    _digest(output["sha256"], "sha256")
    _digest(output["schema_fingerprint"], "schema_fingerprint")
    try:
        connection.execute(
            """
            INSERT INTO job_outputs(
                job_id, ordinal, rows, batches, bytes, sha256,
                schema_fingerprint, relative_path, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                output["ordinal"],
                output["rows"],
                output["batches"],
                output["bytes"],
                output["sha256"],
                output["schema_fingerprint"],
                relative_path,
                timestamp,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise conflict("job output ordinal is already published") from exc


def _transition_updates(updates: dict) -> dict:
    allowed = {
        "selected_device",
        "sealed_at",
        "queued_at",
        "started_at",
        "cancel_requested_at",
        "finished_at",
        "error_code",
        "error_message",
        "result_json",
        "progress_json",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"unsupported job update field(s): {', '.join(sorted(unknown))}")
    encoded = {}
    for key, value in updates.items():
        if key in ("result_json", "progress_json") and value is not None:
            value = _json(value)
        encoded[key] = value
    return encoded


def _decode_row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    for column in tuple(result):
        if column not in _JSON_COLUMNS:
            continue
        value = result.pop(column)
        result[column.removesuffix("_json")] = None if value is None else json.loads(value)
    return result


def _json(value) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_or_none(value) -> str | None:
    return None if value is None else _json(value)


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} must be a canonical UUID")
    return str(parsed)


def _validate_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError("artifact path must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("artifact path must not contain path traversal")


def _nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _now(value: float | None) -> float:
    return time.time() if value is None else float(value)


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
