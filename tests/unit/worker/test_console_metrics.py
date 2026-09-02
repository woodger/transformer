import torch

from app.contracts.worker.v10.config import (
    CheckpointSelectionConfig,
    TrainConfig,
)
from app.contracts.worker.v10.objective import default_objective
from app.worker.telemetry import (
    EpochTelemetry,
    ObservedTrainingEpoch,
    format_epoch_console_line,
)
from app.worker.training.trainer import Trainer


def test_console_line_is_compact_and_human_readable():
    metrics = ObservedTrainingEpoch(
        rows=67249,
        batches=263,
        loss=-3.149016,
        telemetry=EpochTelemetry(
            target_mae={
                "MeanReturn": 0.0225603,
                "SigmaReturn": 0.0231593,
                "ProbTP": 0.12,
                "ProbSL": 0.13,
                "VolatilityNext": 0.14,
                "HittingProbTP": 0.15,
            },
            training_batches_completed=263,
            optimizer_updates_applied=263,
            finite_gradient_batches=263,
            pre_clip_gradient_norm_mean=476.013,
            pre_clip_gradient_norm_max=500.0,
            pre_clip_gradient_norm_p95=490.0,
            elapsed_ms=181677,
        ),
    )

    output = format_epoch_console_line(
        metrics,
        frame=1,
        epoch=2,
        selection_score=3.8270289599977505,
        hidden=256,
        device="cpu",
        checkpoint_best=False,
    )

    assert output == (
        "frame=1 epoch=2 selection=3.82703 loss=-3.149016 "
        "MeanReturn_mae=0.0225603 SigmaReturn_mae=0.0231593 "
        "ProbTP_mae=0.12 ProbSL_mae=0.13 VolatilityNext_mae=0.14 "
        "HittingProbTP_mae=0.15 grad_mean=476.013 rows=67249 "
        "batches=263 time=181.7s"
    )
    assert "hidden=" not in output
    assert "device=" not in output
    assert "checkpoint_best=" not in output


def test_console_line_reports_unavailable_selection_score():
    metrics = ObservedTrainingEpoch()

    assert "selection=n/a" in format_epoch_console_line(
        metrics,
        epoch=1,
        selection_score=None,
    )


def test_trainer_config_line_contains_static_run_configuration():
    trainer = Trainer(
        model=torch.nn.Linear(2, 7),
        device=torch.device("cpu"),
        train_config=TrainConfig(
            lr=5e-4,
            batch_size=256,
            epochs=1,
            selection=CheckpointSelectionConfig(
                min_delta=0.0,
                patience=1,
            ),
        ),
        objective=default_objective(),
        metrics_context={"hidden": 256, "layers": 4, "seq_len": 10},
    )

    assert trainer.config_line() == (
        "config device=cpu batch_size=256 lr=0.0005 hidden=256 layers=4 "
        "seq_len=10 targets=MeanReturn,SigmaReturn,ProbTP,ProbSL,"
        "VolatilityNext,HittingProbTP selection=on context_mode=relaxed "
        "amp=False"
    )
