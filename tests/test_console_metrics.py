import math

import torch

from app.metrics import TrainMetrics
from app.training.trainer import Trainer


def test_console_line_is_compact_and_human_readable():
    metrics = TrainMetrics(
        rows=67249,
        batches=263,
        loss=-3.149016,
        sigma_mean=0.0231593,
        ret_mae=0.0225603,
        ret_mae_baseline=0.00589499,
        ret_mae_skill=3.8270289599977505,
        grad_norm=476.013,
        elapsed_ms=181677,
        loss_stage=1,
    )

    output = metrics.console_line(
        frame=1,
        epoch=2,
        monitor_value=3.8270289599977505,
        max_loss_stage=4,
        hidden=256,
        device="cpu",
        checkpoint_best=False,
    )

    assert output == (
        "frame=1 epoch=2 monitor_value=3.82703 loss=-3.149016 "
        "mae=0.0225603 baseline=0.00589499 skill=3.82703x status=WORSE "
        "sigma=0.0231593 grad=476.013 rows=67249 batches=263 "
        "time=181.7s stage=1/4"
    )
    assert "hidden=" not in output
    assert "device=" not in output
    assert "checkpoint_best=" not in output


def test_console_line_reports_better_and_unavailable_skill():
    better = TrainMetrics(
        ret_mae=0.4,
        ret_mae_baseline=0.5,
        ret_mae_skill=0.8,
        loss_stage=2,
    )
    unavailable = TrainMetrics(
        ret_mae=0.4,
        ret_mae_baseline=0.0,
        ret_mae_skill=math.inf,
        loss_stage=1,
    )

    assert "skill=0.8x status=BETTER" in better.console_line(
        epoch=1,
        monitor_value=0.8,
        max_loss_stage=4,
    )
    assert "skill=n/a status=N/A" in unavailable.console_line(
        epoch=1,
        monitor_value=math.inf,
        max_loss_stage=4,
    )


def test_trainer_config_line_contains_static_run_configuration():
    trainer = Trainer(
        model=torch.nn.Linear(2, 1),
        device=torch.device("cpu"),
        lr=5e-4,
        batch_size=256,
        epochs=1,
        patience=1,
        loss_stage=4,
        stage_size=4,
        metrics_context={"hidden": 256, "layers": 4, "seq_len": 10},
    )

    assert trainer.config_line() == (
        "config device=cpu batch_size=256 lr=0.0005 hidden=256 layers=4 "
        "seq_len=10 loss_schedule=epoch stage_size=4 max_loss_stage=4 "
        "monitor=ret_mae_skill monitor_min_improvement=0 context_mode=relaxed "
        "amp=False"
    )
