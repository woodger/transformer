from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

from app.config import PROJECT_ROOT
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.adapters.outbound.artifact_storage.publication import (
    WorkerArtifactPublisher,
)
from app.service.adapters.outbound.artifact_storage.recovery_publication import (
    RecoveryCheckpointPublisher,
)
from app.service.adapters.outbound.postgres.mapping import execution_job_from_mapping
from app.service.adapters.outbound.worker_probe.device_inventory import (
    static_cuda_inventory,
)
from app.service.adapters.outbound.worker_process.plan import WorkerPlanBuilder
from app.service.adapters.outbound.worker_process.runner import WorkerSubprocessRunner
from app.service.application.ports.devices import DeviceLeaseManager
from app.service.application.services.attempt_executor import (
    WorkerAttemptError,
    WorkerAttemptExecutor,
)
from app.service.application.services.errors import AttemptExecutionError
from app.service.application.services.worker_pool import WorkerPool as WorkerScheduler


class WorkerPool(WorkerScheduler):
    """Compatibility facade and composition for the service scheduler."""

    def __init__(
        self,
        config,
        ledger,
        spool,
        recovery_store=None,
        *,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
        popen_factory: Callable = subprocess.Popen,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
        signal_group: Callable[[int, int], None] = os.killpg,
        python_executable: str | None = None,
        cli_path: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        device_inventory: DeviceLeaseManager | None = None,
        checkpoint_metadata_reader=None,
    ):
        logger = logger or JsonLogger()
        metrics = metrics or OperationalMetrics()
        device_inventory = device_inventory or static_cuda_inventory(lambda: True)
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
        self._popen = popen_factory
        self._argv_hook = argv_hook
        self._signal_group = signal_group
        self._python = python_executable or sys.executable
        self._cli_path = cli_path or os.path.join(PROJECT_ROOT, "app", "main.py")
        self._plan_builder = WorkerPlanBuilder(
            config,
            ledger,
            spool,
            recovery_store,
            python_executable=self._python,
            cli_path=self._cli_path,
        )
        self._artifact_publisher = WorkerArtifactPublisher(
            ledger,
            spool,
            logger=logger,
            metrics=metrics,
            max_payload_bytes=config.max_payload_bytes,
            checkpoint_metadata_reader=checkpoint_metadata_reader,
        )
        self._recovery_publisher = (
            None
            if recovery_store is None
            else RecoveryCheckpointPublisher(
                ledger,
                recovery_store,
                spool,
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
            stage_prediction=self._artifact_publisher.stage_prediction,
            publish_recovery=(
                None
                if self._recovery_publisher is None
                else self._recovery_publisher.publish
            ),
            monotonic=monotonic,
        )
        self.attach_attempt_executor(WorkerAttemptExecutor(
            ledger,
            self._plan_builder,
            self._subprocess_runner,
            self._artifact_publisher,
            logger=logger,
            metrics=metrics,
            argv_hook=argv_hook,
            retry_notifier=self.notify_queued,
            confirm_device_loss=device_inventory.confirm_loss,
            resumable_fit=self._recovery_publisher is not None,
            monotonic=monotonic,
        ))

    def build_argv(self, job: dict, attempt: int | None = None) -> list[str]:
        """Compatibility view of the trusted worker invocation."""

        attempt = job["attempt"] if attempt is None else attempt
        record = execution_job_from_mapping(job)
        try:
            return list(self._plan_builder.build_argv(
                record,
                attempt,
                argv_hook=self._argv_hook,
                job_mapping=job,
            ))
        except AttemptExecutionError as exc:
            raise WorkerAttemptError(exc.code, exc.message) from exc


__all__ = ["WorkerPool"]
