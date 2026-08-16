from __future__ import annotations

import json
import math
import shutil
import subprocess
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from app.contracts.metrics.v2 import (
    build_artifact_document,
    build_training_record,
    project_training_points,
    validate_training_record,
)
from app.project import PROJECT_ROOT

CONTRACT_ROOT = PROJECT_ROOT / "app" / "contracts" / "metrics" / "v2"
JOB_ID = "11111111-1111-4111-8111-111111111111"
ATTEMPT_ID = "22222222-2222-4222-8222-222222222222"
MODEL_REF = "mdl_" + "3" * 32
DEPLOYMENT_ID = "hp800g9.home"


def _interval_metrics() -> dict[str, object]:
    return {
        "mode": "fit-stream",
        "frame": None,
        "epoch": 1,
        "step": 7,
        "rows": 16,
        "batches": 2,
        "lr": 0.001,
        "loss_stage": 4,
        "minimum_loss_stage": 4,
        "maximum_loss_stage": 4,
        "loss": 0.5,
        "loss_l0": 0.1,
        "loss_l1": 0.2,
        "loss_l2": 0.3,
        "loss_l3": 0.4,
        "loss_l4": 0.5,
        "loss_l5": 0.6,
        "loss_nll": 0.7,
        "loss_ev": 0.8,
        "mean_return_mae": 0.01,
        "sigma_return_mae": 0.02,
        "prob_tp_mae": 0.03,
        "prob_sl_mae": 0.04,
        "volatility_next_mae": 0.05,
        "hitting_prob_tp_mae": 0.06,
        "mean_return_rmse": 0.11,
        "sigma_return_rmse": 0.12,
        "prob_tp_rmse": 0.13,
        "prob_sl_rmse": 0.14,
        "volatility_next_rmse": 0.15,
        "hitting_prob_tp_rmse": 0.16,
        "selection_score": None,
        "trainingBatchesCompleted": 2,
        "optimizerUpdatesApplied": 1,
        "optimizerUpdatesSkipped": 1,
        "ampOverflowBatches": 1,
        "finiteGradientBatches": 1,
        "nonFiniteGradientBatches": 1,
        "preClipGradientNormMean": 1.5,
        "preClipGradientNormMax": 1.5,
        "preClipGradientNormP95": 1.5,
        "nan_ratio": 0.0,
        "masked_token_ratio": 0.1,
        "complete_token_ratio": 0.9,
        "partial_token_ratio": 0.1,
        "empty_token_ratio": 0.0,
        "input_pipeline_ms": 1.0,
        "missing_stats_ms": 2.0,
        "host_to_device_ms": 3.0,
        "train_step_ms": 4.0,
        "elapsed_ms": 10.0,
        "checkpoint_best": True,
        "should_stop": False,
        "best_selection_score": 1.2,
    }


def _training_record() -> dict[str, object]:
    return build_training_record(
        _interval_metrics(),
        recorded_at=1_786_809_330.123,
        job_id=JOB_ID,
        attempt_id=ATTEMPT_ID,
        attempt=1,
        model_ref=MODEL_REF,
        data_contract_sha256="a" * 64,
        objective_config_sha256="b" * 64,
        checkpoint_format="transformer-checkpoint-v3",
        application_version="0.1.10",
        git_commit="c" * 40,
    )


def test_metrics_json_schemas_and_opensearch_templates_are_closed():
    for name in (
        "training-record.schema.json",
        "point.schema.json",
        "artifact.schema.json",
    ):
        schema = json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False

    for path in sorted((CONTRACT_ROOT / "opensearch").glob("*.json")):
        template = json.loads(path.read_text(encoding="utf-8"))
        assert "data_stream" not in template
        assert template["template"]["settings"] == {
            "number_of_replicas": 0
        }
        assert template["template"]["mappings"]["dynamic"] == "strict"
        assert template["index_patterns"] == [path.name.removesuffix(
            ".template.json"
        )]


def test_training_record_projects_only_closed_finite_metric_points():
    record = _training_record()
    points = project_training_points(record, deployment_id=DEPLOYMENT_ID)

    names = [point["metric"]["name"] for point in points]
    assert len(names) == len(set(names)) == 41
    assert "training.selection.score" not in names
    assert names[0] == "training.loss.total"
    direct = next(
        point
        for point in points
        if point["metric"]["name"] == "training.loss.direct.mean_return"
    )
    assert direct["targetIndex"] == 0
    assert direct["runId"] == JOB_ID
    assert direct["modelRef"] == MODEL_REF
    assert "frame" not in direct
    assert all(
        isinstance(point["documentSha256"], str)
        and len(point["documentSha256"]) == 64
        for point in points
    )
    gradient_mean = next(
        point
        for point in points
        if point["metric"]["name"]
        == "training.gradient.pre_clip_norm.mean"
    )
    assert gradient_mean["metric"] == {
        "name": "training.gradient.pre_clip_norm.mean",
        "value": 1.5,
        "unit": "1",
    }


def test_nonfinite_or_structurally_extended_training_record_is_rejected():
    record = _training_record()
    nonfinite = deepcopy(record)
    nonfinite["loss"] = math.inf
    with pytest.raises(ValueError, match="must be finite"):
        validate_training_record(nonfinite)

    extended = deepcopy(record)
    extended["arbitrary"] = 1
    with pytest.raises(ValueError, match="Additional properties"):
        validate_training_record(extended)


def test_epoch_batch_and_gradient_invariants_are_enforced():
    inconsistent = _training_record()
    inconsistent["optimizerUpdatesApplied"] = 2

    with pytest.raises(ValueError, match="optimizer counts differ"):
        validate_training_record(inconsistent)

    no_finite_gradients = _training_record()
    no_finite_gradients.update({
        "finiteGradientBatches": 0,
        "nonFiniteGradientBatches": 2,
        "optimizerUpdatesApplied": 0,
        "optimizerUpdatesSkipped": 2,
        "ampOverflowBatches": 2,
        "preClipGradientNormMean": None,
        "preClipGradientNormMax": None,
        "preClipGradientNormP95": None,
    })

    assert validate_training_record(no_finite_gradients) is no_finite_gradients


def test_event_identity_has_a_cross_language_golden_digest():
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the contract test"
    result = subprocess.run(
        [
            node,
            str(CONTRACT_ROOT / "fixtures" / "event_id_sha256.mjs"),
            str(CONTRACT_ROOT / "fixtures" / "event-identity.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    point = project_training_points(
        _training_record(),
        deployment_id=DEPLOYMENT_ID,
    )[0]

    assert result.stdout.strip() == point["eventId"]
    assert point["eventId"] == (
        "f5aafc3d308eb869fba699efe25bc593701805244c32f5d9ef39afa1b653578f"
    )


def test_artifact_document_contains_metadata_but_no_storage_path():
    document = build_artifact_document(
        deployment_id=DEPLOYMENT_ID,
        model_ref=MODEL_REF,
        job_id=JOB_ID,
        attempt_id=ATTEMPT_ID,
        attempt=1,
        application_version="0.1.10",
        git_commit="c" * 40,
        byte_count=1234,
        sha256="d" * 64,
        row_count=3,
        created_at=1_786_809_330.123,
    )

    assert document["schema"] == "inventory.metrics.artifact.v2"
    assert document["runId"] == JOB_ID
    assert "path" not in document
    assert "relativePath" not in document
