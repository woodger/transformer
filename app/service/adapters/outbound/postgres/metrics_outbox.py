from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.service.adapters.outbound.postgres.models import (
    MetricsOutboxEntry,
    ModelMetricsArtifact,
    ModelRunSummaryArtifact,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.domain.records import (
    MetricsOutboxRecord,
    ModelMetricsArtifactRecord,
    ModelRunSummaryArtifactRecord,
)


class PostgresMetricsOutbox:
    """Persist delivery progress without owning training or model lifecycle."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def next_pending(self) -> MetricsOutboxRecord | None:
        now = datetime.now(UTC)
        with self.database.session() as session:
            row = session.execute(
                select(
                    MetricsOutboxEntry,
                    ModelMetricsArtifact,
                    ModelRunSummaryArtifact,
                )
                .join(
                    ModelMetricsArtifact,
                    ModelMetricsArtifact.model_ref
                    == MetricsOutboxEntry.model_ref,
                )
                .outerjoin(
                    ModelRunSummaryArtifact,
                    ModelRunSummaryArtifact.model_ref
                    == MetricsOutboxEntry.model_ref,
                )
                .where(
                    MetricsOutboxEntry.status == "PENDING",
                    MetricsOutboxEntry.next_attempt_at <= now,
                )
                .order_by(MetricsOutboxEntry.created_at)
                .limit(1)
            ).one_or_none()
            if row is None:
                return None
            return _record(row[0], row[1], row[2])

    def advance(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
        cursor: int,
    ) -> bool:
        if cursor <= expected_cursor:
            raise ValueError("outbox cursor must advance")
        with self.database.transaction() as session:
            row = session.get(
                MetricsOutboxEntry,
                model_ref,
                with_for_update=True,
            )
            if row is None or not _owns(row, expected_cursor):
                return False
            row.cursor = cursor
            row.attempts = 0
            row.last_error_code = None
            row.last_error_message = None
            row.updated_at = datetime.now(UTC)
            return True

    def retry(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
        delay_seconds: float,
        error_code: str,
        error_message: str,
    ) -> bool:
        if delay_seconds < 0:
            raise ValueError("outbox retry delay must not be negative")
        now = datetime.now(UTC)
        with self.database.transaction() as session:
            row = session.get(
                MetricsOutboxEntry,
                model_ref,
                with_for_update=True,
            )
            if row is None or not _owns(row, expected_cursor):
                return False
            row.attempts += 1
            row.next_attempt_at = now + timedelta(seconds=delay_seconds)
            row.last_error_code = error_code[:64]
            row.last_error_message = error_message[:1024]
            row.updated_at = now
            return True

    def block(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
        error_code: str,
        error_message: str,
    ) -> bool:
        with self.database.transaction() as session:
            row = session.get(
                MetricsOutboxEntry,
                model_ref,
                with_for_update=True,
            )
            if row is None or not _owns(row, expected_cursor):
                return False
            row.status = "BLOCKED"
            row.last_error_code = error_code[:64]
            row.last_error_message = error_message[:1024]
            row.updated_at = datetime.now(UTC)
            return True

    def complete(
        self,
        model_ref: str,
        *,
        expected_cursor: int,
    ) -> bool:
        now = datetime.now(UTC)
        with self.database.transaction() as session:
            row = session.get(
                MetricsOutboxEntry,
                model_ref,
                with_for_update=True,
            )
            if row is None or not _owns(row, expected_cursor):
                return False
            row.status = "DELIVERED"
            row.attempts = 0
            row.last_error_code = None
            row.last_error_message = None
            row.delivered_at = now
            row.updated_at = now
            return True

    def purge_delivered(self, *, older_than_seconds: float) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        with self.database.transaction() as session:
            rows = session.scalars(
                select(MetricsOutboxEntry).where(
                    MetricsOutboxEntry.status == "DELIVERED",
                    MetricsOutboxEntry.delivered_at < cutoff,
                )
                .with_for_update()
            ).all()
            for row in rows:
                session.delete(row)
            return len(rows)

    def backlog(self) -> tuple[int, int, float | None]:
        now = datetime.now(UTC)
        with self.database.session() as session:
            count, byte_count, oldest = session.execute(
                select(
                    func.count(MetricsOutboxEntry.model_ref),
                    func.coalesce(
                        func.sum(
                            ModelMetricsArtifact.bytes
                            + func.coalesce(ModelRunSummaryArtifact.bytes, 0)
                        ),
                        0,
                    ),
                    func.min(MetricsOutboxEntry.created_at),
                )
                .outerjoin(
                    ModelRunSummaryArtifact,
                    ModelRunSummaryArtifact.model_ref
                    == MetricsOutboxEntry.model_ref,
                )
                .join(
                    ModelMetricsArtifact,
                    ModelMetricsArtifact.model_ref
                    == MetricsOutboxEntry.model_ref,
                )
                .where(
                    MetricsOutboxEntry.status.in_(("PENDING", "BLOCKED"))
                )
            ).one()
        age = None if oldest is None else max(0.0, (now - oldest).total_seconds())
        return int(count), int(byte_count), age


def _owns(row: MetricsOutboxEntry, expected_cursor: int) -> bool:
    return (
        row.status == "PENDING"
        and row.cursor == expected_cursor
    )


def _record(
    outbox: MetricsOutboxEntry,
    artifact: ModelMetricsArtifact,
    run_summary: ModelRunSummaryArtifact | None,
) -> MetricsOutboxRecord:
    return MetricsOutboxRecord(
        artifact=ModelMetricsArtifactRecord(
            model_ref=artifact.model_ref,
            format=artifact.format,
            media_type=artifact.media_type,
            relative_path=artifact.relative_path,
            byte_count=artifact.bytes,
            sha256=artifact.sha256,
            row_count=artifact.row_count,
            job_id=artifact.job_id,
            attempt_id=artifact.attempt_id,
            attempt=artifact.attempt,
            application_version=artifact.application_version,
            git_commit=artifact.git_commit,
            created_at=artifact.created_at.timestamp(),
        ),
        run_summary=(
            None
            if run_summary is None
            else ModelRunSummaryArtifactRecord(
                model_ref=run_summary.model_ref,
                format=run_summary.format,
                media_type=run_summary.media_type,
                relative_path=run_summary.relative_path,
                byte_count=run_summary.bytes,
                sha256=run_summary.sha256,
                job_id=run_summary.job_id,
                attempt_id=run_summary.attempt_id,
                attempt=run_summary.attempt,
                application_version=run_summary.application_version,
                git_commit=run_summary.git_commit,
                created_at=run_summary.created_at.timestamp(),
            )
        ),
        projection_version=outbox.projection_version,
        status=outbox.status,
        cursor=outbox.cursor,
        attempts=outbox.attempts,
        next_attempt_at=outbox.next_attempt_at.timestamp(),
        created_at=outbox.created_at.timestamp(),
    )


__all__ = ["PostgresMetricsOutbox"]
