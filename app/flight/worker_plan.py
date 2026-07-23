from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from collections.abc import Callable, Sequence

from app.flight.constants import ErrorCode
from app.flight.records import ExecutionJobRecord


_COPY_CHUNK_BYTES = 1024 * 1024
_FIT_SPOOL_OPTION = "--input-spool-dir"


@dataclass(frozen=True)
class WorkerPlanError(Exception):
    code: ErrorCode
    message: str

    def __post_init__(self):
        Exception.__init__(self, self.message)


@dataclass(frozen=True)
class ExecutionInput:
    ordinal: int
    rows: int
    byte_count: int
    sha256: str
    absolute_path: str


@dataclass(frozen=True)
class ExecutionPlan:
    inputs: tuple[ExecutionInput, ...]
    argv: tuple[str, ...]
    uses_spooled_fit: bool


class WorkerPlanBuilder:
    """Validate durable artifacts and render one trusted CLI execution plan."""

    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        python_executable: str,
        cli_path: str,
    ):
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.python_executable = python_executable
        self.cli_path = cli_path

    def build(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        *,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
    ) -> ExecutionPlan:
        inputs = self._validated_inputs(job)
        argv = self.build_argv(job, attempt, argv_hook=argv_hook)
        return ExecutionPlan(
            inputs=inputs,
            argv=argv,
            uses_spooled_fit=(
                job.operation == "fit" and _FIT_SPOOL_OPTION in argv
            ),
        )

    def build_argv(
        self,
        job: ExecutionJobRecord,
        attempt: int,
        *,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
        legacy_job: dict | None = None,
    ) -> tuple[str, ...]:
        if not isinstance(attempt, int) or attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        argv = self._default_argv(job, attempt)
        if argv_hook is not None:
            if legacy_job is None:
                legacy_job = self.ledger.get_job(job.job_id)
            if legacy_job is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "claimed job disappeared",
                )
            argv = list(argv_hook(legacy_job, tuple(argv)))
        if not argv or any(
            not isinstance(value, str) or "\x00" in value
            for value in argv
        ):
            raise ValueError("worker argv hook returned invalid argv")
        return tuple(argv)

    def _validated_inputs(
        self,
        job: ExecutionJobRecord,
    ) -> tuple[ExecutionInput, ...]:
        inputs = self.ledger.list_committed_inputs(job.job_id)
        if [item.ordinal for item in inputs] != list(range(len(inputs))):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "sealed input ordinals are inconsistent",
            )
        prepared = []
        for item in inputs:
            expected = self.spool.input_path(job.job_id, item.ordinal)
            actual = self.spool.absolute_path(item.relative_path)
            if actual != expected:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "committed input path is invalid",
                )
            try:
                size = os.path.getsize(actual)
            except OSError as exc:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "committed input is unavailable",
                ) from exc
            if size != item.byte_count or size > self.config.max_payload_bytes:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "committed input size is invalid",
                )
            if _sha256_file(actual) != item.sha256:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "committed input digest is invalid",
                )
            prepared.append(ExecutionInput(
                ordinal=item.ordinal,
                rows=item.rows,
                byte_count=item.byte_count,
                sha256=item.sha256,
                absolute_path=actual,
            ))
        return tuple(prepared)

    def _default_argv(
        self,
        job: ExecutionJobRecord,
        attempt: int,
    ) -> list[str]:
        device = job.selected_device
        if device not in ("cpu", "cuda"):
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "queued job has no selected device",
            )
        base = [self.python_executable, self.cli_path]
        if job.operation == "predict":
            model = self.ledger.get_model_artifact(
                job.input_model_ref,
                owner_subject=job.owner_subject,
            )
            if model is None:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "resolved model generation is unavailable",
                )
            checkpoint = self.spool.model_absolute_path(
                model.checkpoint_path
            )
            expected = self.spool.model_checkpoint_path(model.model_ref)
            if checkpoint != expected or not os.path.isfile(checkpoint):
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "resolved model checkpoint is unavailable",
                )
            if _sha256_file(checkpoint) != model.sha256:
                raise WorkerPlanError(
                    ErrorCode.INTERNAL,
                    "resolved model checkpoint digest is invalid",
                )
            return [
                *base,
                "predict-stream",
                "--device", device,
                "--checkpoint", checkpoint,
                "--pred-col", job.prediction_column,
                "--max-frame-bytes", str(self.config.max_payload_bytes),
            ]

        model = job.model_config
        train = job.training_config
        if model is None or train is None:
            raise WorkerPlanError(
                ErrorCode.INTERNAL,
                "fit job configuration is unavailable",
            )
        argv = [
            *base,
            "fit-stream",
            "--device", device,
            "--checkpoint-out", self.spool.attempt_checkpoint_path(
                job.job_id,
                attempt,
            ),
            "--metrics-out", self.spool.attempt_metrics_path(
                job.job_id,
                attempt,
            ),
            "--max-frame-bytes", str(self.config.max_payload_bytes),
            _FIT_SPOOL_OPTION, self.spool.input_directory(job.job_id),
            "--input-frame-count", str(job.input_frame_count),
            "--seq-len", str(model.seq_len),
            "--hidden", str(model.hidden),
            "--layers", str(model.layers),
            "--dropout", str(model.dropout),
            "--nhead", str(model.nhead),
            "--mode", model.context_mode,
            "--lr", str(train.lr),
            "--weight-decay", str(train.weight_decay),
            "--batch-size", str(train.batch_size),
            "--epochs", str(train.epochs),
            "--loss-stage", str(train.loss_stage),
            "--loss-schedule", train.loss_schedule,
            "--stage-size", str(train.stage_size),
            "--patience", str(train.patience),
            "--monitor", train.monitor,
            "--monitor-min-improvement", str(
                train.monitor_min_improvement
            ),
            "--seed", str(train.seed),
            (
                "--save-best-checkpoint"
                if train.save_best_checkpoint
                else "--no-save-best-checkpoint"
            ),
        ]
        if train.use_amp:
            argv.append("--use-amp")
        if train.deterministic:
            argv.append("--deterministic")
        return argv


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
