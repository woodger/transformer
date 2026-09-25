from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

from sqlalchemy.orm import Session

from app.contracts.metrics.fit_run.v9 import PROJECTION_VERSION
from app.service.adapters.outbound.postgres.models import MetricsOutboxEntry
from app.service.adapters.outbound.postgres.session import Database
from app.service.adapters.outbound.postgres.telemetry.repository import (
    PostgresTrainingTelemetry,
)

_JOB_ID = "00000000-0000-4000-8000-000000000001"
_ATTEMPT_ID = "00000000-0000-4000-8000-000000000002"
_MODEL_REF = "mdl_00000000000000000000000000000001"


def test_register_run_artifacts_uses_current_metrics_projection_version():
    added: list[object] = []
    session = cast(
        Session,
        SimpleNamespace(
            get=lambda *_args, **_kwargs: SimpleNamespace(
                producing_job_id=_JOB_ID,
            ),
            scalar=lambda _statement: None,
            execute=lambda _statement: SimpleNamespace(one=lambda: (0, 0)),
            add=added.append,
            flush=lambda: None,
        ),
    )
    database = cast(
        Database,
        SimpleNamespace(transaction=lambda: nullcontext(session)),
    )

    registered = PostgresTrainingTelemetry(database).register_run_artifacts(
        model_ref=_MODEL_REF,
        job_id=_JOB_ID,
        attempt_id=_ATTEMPT_ID,
        attempt=1,
        metrics_path="jobs/test/metrics.jsonl",
        metrics_format="transformer.training-metrics.v9",
        metrics_media_type="application/x-ndjson",
        metrics_byte_count=1,
        metrics_sha256="a" * 64,
        metrics_row_count=1,
        run_summary_path="jobs/test/run-summary.json",
        run_summary_format="transformer.fit-run-summary.v9",
        run_summary_media_type="application/json",
        run_summary_byte_count=1,
        run_summary_sha256="b" * 64,
        application_version="0.1.0",
        git_commit="c" * 40,
        max_outbox_entries=1,
        max_outbox_bytes=2,
        now=1.0,
    )

    outbox = next(item for item in added if isinstance(item, MetricsOutboxEntry))

    assert registered is True
    assert outbox.projection_version == PROJECTION_VERSION
