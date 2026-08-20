import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from app.contracts.worker.v7.config import CheckpointSelectionConfig
from app.contracts.worker.v7.objective import objective_config_sha256
from app.worker.checkpoints.recovery import (
    load_training_recovery,
    save_training_recovery,
)
from app.worker.data.tensors import TrainingBatch
from app.worker.training.run_config import ModelConfig, TrainConfig
from app.worker.training.trainer import Trainer

CONFIG_HASH = "a" * 64
MANIFEST_HASH = "b" * 64


class InjectedInterruption(Exception):
    pass


def _model() -> nn.Module:
    class TargetAlignedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4, 12),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(12, 7),
            )

        def forward(self, value):
            raw = self.layers(value)
            return torch.stack((
                torch.tanh(raw[:, 0]),
                torch.sigmoid(raw[:, 1]),
                raw[:, 2],
                raw[:, 3],
                torch.sigmoid(raw[:, 4]),
                raw[:, 5],
                torch.nn.functional.softplus(raw[:, 6]) + 1e-6,
            ), dim=1)

    return TargetAlignedLinear()


def _trainer(initial_state: dict) -> Trainer:
    model = _model()
    model.load_state_dict(initial_state)
    model_config = ModelConfig(
        seq_len=2,
        hidden=8,
        layers=1,
        dropout=0.2,
        nhead=2,
        feature_dim=2,
    )
    train_config = TrainConfig(
        lr=0.001,
        batch_size=3,
        epochs=3,
        loss_stage=4,
        loss_schedule="none",
        stage_size=2,
        selection=CheckpointSelectionConfig(min_delta=0.0, patience=0),
        seed=919,
        deterministic=True,
    )
    return Trainer(
        model=model,
        device=torch.device("cpu"),
        train_config=train_config,
        model_config=model_config,
    )


def _payloads(features: torch.Tensor, targets: torch.Tensor):
    def open_payloads():
        yield TrainingBatch(
            features=features[:5].clone(),
            targets=targets[:5].clone(),
        )
        yield TrainingBatch(
            features=features[5:].clone(),
            targets=targets[5:].clone(),
        )

    return open_payloads


def _seed() -> None:
    random.seed(731)
    np.random.seed(731)
    torch.manual_seed(731)


def _assert_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_tree_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tree_equal(left_item, right_item)
        return
    assert left == right


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    _seed()
    source = torch.randn(11, 2, 2)
    target = torch.rand(11, 6)
    target[:, 0] = torch.rand(11) * 2 - 1
    initial_state = copy.deepcopy(_model().state_dict())
    payloads = _payloads(source, target)

    _seed()
    uninterrupted = _trainer(initial_state)
    uninterrupted.fit_payloads_resumable(payloads)

    checkpoint = tmp_path / "1.pth"
    _seed()
    interrupted = _trainer(initial_state)

    def stop_after_first_epoch(*_args):
        event = save_training_recovery(
            str(checkpoint),
            interrupted,
            generation=interrupted.state.global_epoch,
            config_hash=CONFIG_HASH,
            manifest_hash=MANIFEST_HASH,
        )
        assert event["completed_epochs"] == 1
        raise InjectedInterruption

    with pytest.raises(InjectedInterruption):
        interrupted.fit_payloads_resumable(
            payloads,
            on_epoch_committed=stop_after_first_epoch,
        )

    payload = load_training_recovery(
        str(checkpoint),
        torch.device("cpu"),
        expected_config_hash=CONFIG_HASH,
        expected_manifest_hash=MANIFEST_HASH,
        expected_objective_config_sha256=objective_config_sha256(
            interrupted.train_config
        ),
    )
    resumed = _trainer(initial_state)
    resumed.load_recovery_state_dict(payload["trainer_state"])
    resumed.fit_payloads_resumable(payloads)

    assert resumed.state == uninterrupted.state
    assert resumed.selection_state == uninterrupted.selection_state
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.best_selection_score == uninterrupted.best_selection_score
    _assert_tree_equal(
        resumed.model.state_dict(),
        uninterrupted.model.state_dict(),
    )
    _assert_tree_equal(
        resumed.optimizer.state_dict(),
        uninterrupted.optimizer.state_dict(),
    )
    _assert_tree_equal(
        resumed.best_state_dict,
        uninterrupted.best_state_dict,
    )


def test_recovery_checkpoint_rejects_a_different_closed_input_set(
    tmp_path,
):
    _seed()
    source = torch.randn(3, 2, 2)
    target = torch.rand(3, 6)
    target[:, 0] = torch.rand(3) * 2 - 1
    initial_state = copy.deepcopy(_model().state_dict())
    trainer = _trainer(initial_state)
    trainer.fit_payloads_resumable(_payloads(source, target))
    checkpoint = tmp_path / "3.pth"
    save_training_recovery(
        str(checkpoint),
        trainer,
        generation=trainer.state.global_epoch,
        config_hash=CONFIG_HASH,
        manifest_hash=MANIFEST_HASH,
    )

    with pytest.raises(ValueError, match="closed job"):
        load_training_recovery(
            str(checkpoint),
            torch.device("cpu"),
            expected_config_hash=CONFIG_HASH,
            expected_manifest_hash="c" * 64,
            expected_objective_config_sha256=objective_config_sha256(
                trainer.train_config
            ),
        )
