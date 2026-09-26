import os

import pytest
import torch

import app.worker.checkpoints.model as checkpoint_module
from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v18.config import TrainConfig
from app.contracts.worker.v18.model_config import ModelConfig
from app.contracts.worker.v18.model_definition import resolved_semantic_digests
from app.worker.checkpoints.model import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    model_path,
    save_checkpoint,
)
from tests.fixture_documents import semantic_fixture_document


def _checkpoint_metadata():
    document = semantic_fixture_document("single-regression")["modelContract"]
    assert isinstance(document, dict)
    tuning = document["modelTuning"]
    assert isinstance(tuning, dict)
    tuning.update({
        "hiddenWidth": 8,
        "encoderLayerCount": 1,
        "dropoutProbability": 0.0,
        "attentionHeadCount": 2,
    })
    contract = ModelContract.from_document(document)
    model_config = ModelConfig.from_tuning(
        contract.model_tuning,
        seq_len=1,
        feature_dim=2,
    )
    digests = resolved_semantic_digests(
        contract,
        "a" * 64,
        model_config,
    )
    return {
        "format": CHECKPOINT_FORMAT,
        "serviceVersion": "0.2.0",
        "generation": 1,
        "jobId": "11111111-1111-4111-8111-111111111111",
        "dataContract": {
            "dataContractSha256": "a" * 64,
            "seqLen": 1,
            "featureDim": 2,
        },
        "modelContract": contract.to_document(),
        "predictionDefinition": contract.prediction_definition(seq_len=1),
        "modelConfig": {
            "seqLen": 1,
            "featureDim": 2,
            "hiddenWidth": 8,
            "encoderLayerCount": 1,
            "dropoutProbability": 0.0,
            "attentionHeadCount": 2,
            "missingValuePolicy": "relaxed",
        },
        "semanticDigests": digests,
        "trainingConfig": TrainConfig().to_manifest(),
        "diagnostics": TrainConfig().diagnostics.to_document(),
        "selection": {
            "enabled": False,
                "modelDefinitionSha256": digests["modelDefinitionSha256"],
            "bestSelectionScore": None,
            "bestEpoch": None,
            "source": "last_epoch",
        },
        "initialization": {"source": "random"},
        "jobConfigSha256": "b" * 64,
        "manifestSha256": "c" * 64,
        "progress": {
            "completedEpochs": 1,
            "globalStep": 1,
            "trainingComplete": True,
        },
    }


def test_relative_model_path_cannot_escape_models_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint_module, "MODELS_DIR", str(tmp_path / "models"))

    with pytest.raises(ValueError, match="must stay inside models"):
        model_path("../outside.pth")


def test_absolute_model_path_remains_supported(tmp_path):
    path = tmp_path / "nested" / "model.pth"

    assert model_path(path) == str(path)


def test_relative_model_path_cannot_escape_through_symlink(monkeypatch, tmp_path):
    models = tmp_path / "models"
    outside = tmp_path / "outside"
    models.mkdir()
    outside.mkdir()
    (models / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(checkpoint_module, "MODELS_DIR", str(models))

    with pytest.raises(ValueError, match="must stay inside models"):
        model_path("linked/model.pth")


def test_checkpoint_save_atomically_replaces_existing_file(tmp_path):
    path = tmp_path / "nested" / "model.pth"
    model = torch.nn.Linear(2, 1)
    path.parent.mkdir()
    path.write_bytes(b"old checkpoint")

    save_checkpoint(
        path,
        model,
        metadata=_checkpoint_metadata(),
    )

    checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["metadata"]["format"] == CHECKPOINT_FORMAT
    assert set(checkpoint["state_dict"]) == {"weight", "bias"}
    assert not [name for name in os.listdir(path.parent) if name.endswith(".tmp")]


def test_unknown_wrapped_checkpoint_format_is_rejected(tmp_path):
    path = tmp_path / "model.pth"
    torch.save(
        {
            "metadata": {"format": "transformer-checkpoint-v999"},
            "state_dict": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(path, torch.device("cpu"))


@pytest.mark.parametrize("corruption", ["data-digest", "target-contract"])
def test_checkpoint_rejects_inconsistent_embedded_semantics(
    tmp_path,
    corruption,
):
    path = tmp_path / "model.pth"
    save_checkpoint(
        path,
        torch.nn.Linear(2, 1),
        metadata=_checkpoint_metadata(),
    )
    payload = torch.load(path, weights_only=False)
    if corruption == "data-digest":
        payload["metadata"]["dataContract"]["dataContractSha256"] = "d" * 64
    else:
        del payload["metadata"]["modelContract"]["targetContract"]["slots"][0][
            "observedConstraint"
        ]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint contents are invalid"):
        load_checkpoint(path, torch.device("cpu"))
