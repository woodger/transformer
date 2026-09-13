from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

import app.service.adapters.outbound.artifacts.recovery_publication as publication_module
from app.contracts.checkpoint.v7 import CHECKPOINT_FORMAT, RECOVERY_FORMAT
from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.artifacts.recovery_publication import (
    RecoveryCheckpointPublisher,
)
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import (
    ExecutionJobRecord,
    TrainingRecoveryCheckpointRecord,
)
from tests.support.consumer_neutral import model_contract


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
        }
        return TrainingRecoveryCheckpointRecord(
            job_id=fields["job_id"],
            generation=fields["generation"],
            attempt=fields["attempt"],
            input_revision=fields["input_revision"],
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
    contract = model_contract(
        "single-regression",
        seq_len=2,
        feature_dim=2,
    )
    data_contract = {
        "identity": "test.dataset",
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": "d" * 64,
        "seqLen": 2,
        "featureDim": 2,
    }
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
        source_encoding={
            "encoding": "indexedFeatureBlocks",
            "featureBlocks": [
                {"position": 0, "windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        model_config=None,
        training_config=None,
        data_contract=data_contract,
        model_contract=contract.to_document(),
        semantic_digests=contract.digests("d" * 64),
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
    spool = Spool(
        str(tmp_path / "runtime"),
        str(tmp_path / "models"),
        str(tmp_path / "telemetry"),
    ).initialize()
    checkpoint_path = spool.attempt_recovery_checkpoint_path(job_id, 1, 1)
    spool.ensure_parent(checkpoint_path)
    checkpoint = b"recovery checkpoint"
    Path(checkpoint_path).write_bytes(checkpoint)
    ledger = _Ledger(job)
    telemetry = _Telemetry()
    logger = _Logger()
    monkeypatch.setattr(
        publication_module,
        "validate_training_metrics_for_model",
        lambda document, _contract: document,
    )
    operational_metrics = OperationalMetrics()
    publisher = RecoveryCheckpointPublisher(
        ledger,
        recovery_store,
        spool,
        telemetry=telemetry,
        logger=logger,
        metrics=operational_metrics,
    )

    publisher.publish(job, {
        "format": RECOVERY_FORMAT,
        "generation": 1,
        "progress": {
            "completedEpochs": 1,
            "globalStep": 2,
            "trainingComplete": False,
        },
        "artifact": {
            "path": checkpoint_path,
            "format": CHECKPOINT_FORMAT,
            "byteCount": len(checkpoint),
            "checkpointSha256": hashlib.sha256(checkpoint).hexdigest(),
        },
        "checkpointSerializationMs": 1.0,
        "metrics": {"epoch": 1, "step": 2},
    })

    assert ledger.checkpoints == 1
    assert ledger.progress == {
        "epoch": 1,
        "step": 2,
    }
    assert telemetry.metric_intervals == 1
    assert not os.path.exists(checkpoint_path)
    published_path = recovery_store.checkpoint_path(job_id, 1)
    assert Path(published_path).read_bytes() == checkpoint
    counters = operational_metrics.snapshot()["counters"]
    assert counters["recoveryCheckpointStagingRemoved"] == 1
    assert counters["recoveryCheckpointStagingBytesRemoved"] == len(checkpoint)
    assert any(
        event == "metrics.collection.failed"
        and fields["phase"] == "recovery-checkpoint"
        for event, fields in logger.events
    )


def test_registration_failure_preserves_worker_checkpoint_staging(
    tmp_path,
):
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    contract = model_contract(
        "single-regression",
        seq_len=2,
        feature_dim=2,
    )
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
        source_encoding={
            "encoding": "indexedFeatureBlocks",
            "featureBlocks": [
                {"position": 0, "windowRows": 1, "nativeRowWidth": 2},
            ],
        },
        model_config=None,
        training_config=None,
        data_contract={
            "identity": "test.dataset",
            "revision": 1,
            "profile": "test.profile",
            "dataContractSha256": "d" * 64,
            "seqLen": 2,
            "featureDim": 2,
        },
        model_contract=contract.to_document(),
        semantic_digests=contract.digests("d" * 64),
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

    class FailingLedger(_Ledger):
        def register_recovery_checkpoint(self, **_fields):
            raise RuntimeError("injected registration failure")

    recovery_store = RecoveryStore(str(tmp_path / "recovery")).initialize()
    spool = Spool(
        str(tmp_path / "runtime"),
        str(tmp_path / "models"),
        str(tmp_path / "telemetry"),
    ).initialize()
    checkpoint_path = spool.attempt_recovery_checkpoint_path(job_id, 1, 1)
    spool.ensure_parent(checkpoint_path)
    checkpoint = b"recovery checkpoint"
    Path(checkpoint_path).write_bytes(checkpoint)
    publisher = RecoveryCheckpointPublisher(
        FailingLedger(job),
        recovery_store,
        spool,
        logger=_Logger(),
        metrics=OperationalMetrics(),
    )

    with pytest.raises(RuntimeError, match="injected registration failure"):
        publisher.publish(job, {
            "format": RECOVERY_FORMAT,
            "generation": 1,
            "progress": {
                "completedEpochs": 1,
                "globalStep": 2,
                "trainingComplete": False,
            },
            "artifact": {
                "path": checkpoint_path,
                "format": CHECKPOINT_FORMAT,
                "byteCount": len(checkpoint),
                "checkpointSha256": hashlib.sha256(checkpoint).hexdigest(),
            },
        })

    assert Path(checkpoint_path).read_bytes() == checkpoint
