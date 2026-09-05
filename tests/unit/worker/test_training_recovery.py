import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from app.contracts.worker.v12.config import CheckpointSelectionConfig
from app.worker.application.artifacts import checkpoint_metadata
from app.worker.checkpoints.recovery import (
    load_training_recovery,
    save_training_recovery,
)
from app.worker.data.tensors import TrainingBatch
from app.worker.training.run_config import ModelConfig, TrainConfig
from app.worker.training.trainer import Trainer
from tests.support.consumer_neutral import data_contract, model_contract

CONFIG_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
MODEL_CONTRACT = model_contract(
    "multi-target-shared-resource",
    seq_len=2,
    feature_dim=2,
    hidden=8,
    layers=1,
    dropout=0.2,
    nhead=2,
    mode="relaxed",
)


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
                nn.Linear(12, 4),
            )

        def forward(self, value):
            return self.layers(value)

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
        selection=CheckpointSelectionConfig(min_delta=0.0, patience=0),
        seed=919,
        deterministic=True,
    )
    return Trainer(
        model=model,
        device=torch.device("cpu"),
        train_config=train_config,
        model_contract=MODEL_CONTRACT,
        model_config=model_config,
        data_contract=data_contract(MODEL_CONTRACT),
        initialization={"kind": "random"},
    )


def _manifest() -> dict:
    return {
        "jobId": "11111111-1111-4111-8111-111111111111",
        "dataContract": data_contract(MODEL_CONTRACT),
        "modelContract": MODEL_CONTRACT.to_document(),
        "semanticDigests": MODEL_CONTRACT.digests("d" * 64),
        "jobConfigSha256": CONFIG_HASH,
        "manifestSha256": MANIFEST_HASH,
    }


def _descriptor(metadata: dict) -> dict:
    return {
        key: metadata[key]
        for key in (
            "jobId",
            "generation",
            "jobConfigSha256",
            "semanticDigests",
            "manifestSha256",
            "progress",
        )
    }


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
    target = torch.rand(11, MODEL_CONTRACT.target_width)
    initial_state = copy.deepcopy(_model().state_dict())
    payloads = _payloads(source, target)

    _seed()
    uninterrupted = _trainer(initial_state)
    uninterrupted.fit_payloads_resumable(payloads)

    checkpoint = tmp_path / "1.pth"
    _seed()
    interrupted = _trainer(initial_state)
    saved_metadata = None

    def stop_after_first_epoch(*_args):
        nonlocal saved_metadata
        saved_metadata = checkpoint_metadata(interrupted, _manifest())
        event = save_training_recovery(
            str(checkpoint),
            interrupted,
            metadata=saved_metadata,
        )
        assert event["completedEpochs"] == 1
        raise InjectedInterruption

    with pytest.raises(InjectedInterruption):
        interrupted.fit_payloads_resumable(
            payloads,
            on_epoch_committed=stop_after_first_epoch,
        )

    assert saved_metadata is not None
    payload = load_training_recovery(
        str(checkpoint),
        torch.device("cpu"),
        descriptor=_descriptor(saved_metadata),
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
    target = torch.rand(3, MODEL_CONTRACT.target_width)
    initial_state = copy.deepcopy(_model().state_dict())
    trainer = _trainer(initial_state)
    trainer.fit_payloads_resumable(_payloads(source, target))
    checkpoint = tmp_path / "3.pth"
    metadata = checkpoint_metadata(trainer, _manifest())
    save_training_recovery(
        str(checkpoint),
        trainer,
        metadata=metadata,
    )

    descriptor = _descriptor(metadata)
    descriptor["manifestSha256"] = "c" * 64
    with pytest.raises(ValueError, match="fence"):
        load_training_recovery(
            str(checkpoint),
            torch.device("cpu"),
            descriptor=descriptor,
        )
