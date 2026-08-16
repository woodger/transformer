import torch

from app.contracts.worker.v4.config import (
    CheckpointSelectionConfig,
    TrainConfig,
)
from app.worker.metrics import TrainMetrics
from app.worker.training.trainer import Trainer


def test_console_line_is_compact_and_human_readable():
    metrics = TrainMetrics(
        rows=67249,
        batches=263,
        loss=-3.149016,
        mean_return_mae=0.0225603,
        sigma_return_mae=0.0231593,
        prob_tp_mae=0.12,
        prob_sl_mae=0.13,
        volatility_next_mae=0.14,
        hitting_prob_tp_mae=0.15,
        grad_norm=476.013,
        elapsed_ms=181677,
        loss_stage=1,
    )

    output = metrics.console_line(
        frame=1,
        epoch=2,
        selection_score=3.8270289599977505,
        max_loss_stage=4,
        hidden=256,
        device="cpu",
        checkpoint_best=False,
    )

    assert output == (
        "frame=1 epoch=2 selection=3.82703 loss=-3.149016 "
        "mean_mae=0.0225603 sigma_mae=0.0231593 tp_mae=0.12 "
        "sl_mae=0.13 vol_mae=0.14 hit_mae=0.15 grad=476.013 "
        "rows=67249 batches=263 time=181.7s stage=1/4"
    )
    assert "hidden=" not in output
    assert "device=" not in output
    assert "checkpoint_best=" not in output


def test_console_line_reports_unavailable_selection_score():
    metrics = TrainMetrics(loss_stage=1)

    assert "selection=n/a" in metrics.console_line(
        epoch=1,
        selection_score=None,
        max_loss_stage=4,
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
            loss_stage=4,
            stage_size=4,
        ),
        metrics_context={"hidden": 256, "layers": 4, "seq_len": 10},
    )

    assert trainer.config_line() == (
        "config device=cpu batch_size=256 lr=0.0005 hidden=256 layers=4 "
        "seq_len=10 loss_schedule=epoch stage_size=4 max_loss_stage=4 "
        "selection=on context_mode=relaxed "
        "amp=False"
    )
