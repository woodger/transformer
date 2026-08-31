from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import app.service.adapters.outbound.artifacts.recovery_publication as publication_module
from app.contracts.worker.v8.objective import TRAINING_RECOVERY_FORMAT
from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.artifacts.recovery_publication import (
    RecoveryCheckpointPublisher,
)
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import (
    ExecutionJobRecord,
    TrainingRecoveryCheckpointRecord,
)


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class _Ledger:
    def __init__(self, job: ExecutionJobRecord) -> None:
        self.job = job
        self.checkpoints = 0
        self.progress: dict[str, object] | None = None

    def get_execution_job(self, job_id: str) -> ExecutionJobRecord | None:
        return self.job if job_id == self.job.job_id else None

    def register_recovery_checkpoint(self, **fields):
        self.checkpoints += 1
        self.progress = {
            "epoch": fields["completed_epochs"],
            "step": fields["global_step"],
            "loss": fields["loss"],
        }
        return TrainingRecoveryCheckpointRecord(
            job_id=fields["job_id"],
            generation=fields["generation"],
            attempt=fields["attempt"],
            format=fields["format"],
            relative_path=fields["relative_path"],
            byte_count=fields["byte_count"],
            sha256=fields["sha256"],
            completed_epochs=fields["completed_epochs"],
            global_step=fields["global_step"],
            training_complete=fields["training_complete"],
        ), False

    def prune_recovery_checkpoints(
        self,
        _job_id: str,
        *,
        keep: int = 2,
    ) -> list[str]:
        assert keep == 2
        return []


class _Telemetry:
    def __init__(self) -> None:
        self.metric_intervals = 0

    def record_epoch_interval(self, **_fields):
        self.metric_intervals += 1
        raise RuntimeError("injected telemetry persistence failure")


def test_metric_persistence_failure_does_not_reject_recovery_checkpoint(
    tmp_path,
    monkeypatch,
):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    job = ExecutionJobRecord(
        job_id=job_id,
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.CLOSED,
        execution_state=ExecutionState.RUNNING,
        input_revision=1,
        selected_device="cpu",
        model_label="daily",
        input_model_ref=None,
        prediction_column="predictions",
        model_config=None,
        training_config=None,
        data_contract={},
        ml_contract={},
        config_hash="a" * 64,
        manifest_sha256="b" * 64,
        feature_dim=2,
        input_frame_count=1,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
        attempt_id=attempt_id,
    )
    recovery_store = RecoveryStore(str(tmp_path / "recovery")).initialize()
    checkpoint_path = recovery_store.checkpoint_path(job_id, 1)
    recovery_store.ensure_parent(checkpoint_path)
    checkpoint = b"recovery checkpoint"
    Path(checkpoint_path).write_bytes(checkpoint)
    ledger = _Ledger(job)
    telemetry = _Telemetry()
    logger = _Logger()
    monkeypatch.setattr(
        publication_module,
        "validate_document",
        lambda document, _schema: document,
    )
    publisher = RecoveryCheckpointPublisher(
        ledger,
        recovery_store,
        telemetry=telemetry,
        logger=logger,
        metrics=OperationalMetrics(),
    )

    publisher.publish(job, {
        "format": TRAINING_RECOVERY_FORMAT,
        "generation": 1,
        "completed_epochs": 1,
        "global_step": 2,
        "progress": {
            "epoch": 1,
            "step": 2,
            "loss": -3.149016,
        },
        "training_complete": False,
        "bytes": len(checkpoint),
        "sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_serialization_ms": 1.0,
        "checkpoint_publication_ms": 2.0,
        "metrics": {"epoch": 1, "step": 2},
    })

    assert ledger.checkpoints == 1
    assert ledger.progress == {
        "epoch": 1,
        "step": 2,
        "loss": -3.149016,
    }
    assert telemetry.metric_intervals == 1
    assert any(
        event == "metrics.collection.failed"
        and fields["phase"] == "recovery-persistence"
        for event, fields in logger.events
    )
