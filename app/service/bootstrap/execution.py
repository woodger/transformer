from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable

from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.artifacts.publication import (
    WorkerArtifactPublisher,
)
from app.service.adapters.outbound.artifacts.recovery_publication import (
    RecoveryCheckpointPublisher,
)
from app.service.adapters.outbound.artifacts.recovery_store import RecoveryStore
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.adapters.outbound.artifacts.telemetry.publication import (
    FitRunTelemetryPublisher,
)
from app.service.adapters.outbound.cuda.inventory import (
    static_cuda_inventory,
)
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.worker.plan import WorkerPlanBuilder
from app.service.adapters.outbound.worker.runner import (
    PopenFactory,
    WorkerSubprocessRunner,
)
from app.service.application.ports.devices import DeviceLeaseManager
from app.service.application.ports.telemetry import TrainingTelemetryRepository
from app.service.application.services.attempt_executor import (
    WorkerAttemptExecutor,
)
from app.service.application.services.worker_pool import WorkerPool as WorkerScheduler
from app.service.bootstrap.build_identity import BuildIdentity, load_build_identity
from app.service.bootstrap.config import FlightServiceConfig


class WorkerPool(WorkerScheduler):
    """Собрать долговременный планировщик с процессными адаптерами Worker v20."""

    def __init__(
        self,
        config: FlightServiceConfig,
        ledger: Ledger,
        spool: Spool,
        recovery_store: RecoveryStore | None = None,
        *,
        telemetry: TrainingTelemetryRepository | None = None,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
        signal_group: Callable[[int, int], None] = os.killpg,
        python_executable: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        device_inventory: DeviceLeaseManager | None = None,
        build_identity: BuildIdentity | None = None,
    ) -> None:
        logger = logger or JsonLogger()
        metrics = metrics or OperationalMetrics()
        device_inventory = device_inventory or static_cuda_inventory(lambda: True)
        build_identity = build_identity or load_build_identity()
        super().__init__(
            config,
            ledger,
            logger=logger,
            metrics=metrics,
            device_inventory=device_inventory,
            monotonic=monotonic,
        )
        self.spool = spool
        self.recovery_store = recovery_store
        self._python = python_executable or sys.executable
        self._plan_builder = WorkerPlanBuilder(
            config,
            ledger,
            spool,
            recovery_store,
            python_executable=self._python,
        )
        self._artifact_publisher = WorkerArtifactPublisher(
            ledger,
            spool,
            logger=logger,
            metrics=metrics,
            max_payload_bytes=config.max_payload_bytes,
        )
        self._fit_telemetry_publisher = (
            None
            if telemetry is None
            else FitRunTelemetryPublisher(
                telemetry,
                spool,
                logger=logger,
                metrics=metrics,
                application_version=build_identity.application_version,
                git_commit=build_identity.git_commit,
            )
        )
        self._recovery_publisher = (
            None
            if recovery_store is None
            else RecoveryCheckpointPublisher(
                ledger,
                recovery_store,
                spool,
                telemetry=telemetry,
                logger=logger,
                metrics=metrics,
            )
        )
        self._subprocess_runner = WorkerSubprocessRunner(
            config,
            ledger,
            spool,
            logger=logger,
            popen_factory=popen_factory,
            signal_group=signal_group,
            python_executable=self._python,
            publish_recovery=(
                None
                if self._recovery_publisher is None
                else self._recovery_publisher.publish
            ),
            stream_inputs=self._plan_builder.streaming_inputs,
            monotonic=monotonic,
        )
        self.attach_attempt_executor(WorkerAttemptExecutor(
            ledger,
            self._plan_builder,
            self._subprocess_runner,
            self._artifact_publisher,
            fit_telemetry_publisher=self._fit_telemetry_publisher,
            logger=logger,
            metrics=metrics,
            retry_notifier=self.notify_queued,
            confirm_device_loss=device_inventory.confirm_loss,
            resumable_fit=self._recovery_publisher is not None,
            monotonic=monotonic,
        ))

__all__ = ["WorkerPool"]
