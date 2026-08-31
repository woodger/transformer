from __future__ import annotations

import json
import math
import shutil
import subprocess
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from app.contracts.metrics.fit_run.v3 import (
    build_run_document,
    build_run_summary,
    validate_run_summary,
)
from app.project import PROJECT_ROOT

CONTRACT_ROOT = (
    PROJECT_ROOT / "app" / "contracts" / "metrics" / "fit_run" / "v3"
)
JOB_ID = "11111111-1111-4111-8111-111111111111"
ATTEMPT_ID = "22222222-2222-4222-8222-222222222222"
MODEL_REF = "mdl_" + "3" * 32
def _summary():
    return build_run_summary(
        recorded_at=1_786_809_330.123,
        job_id=JOB_ID,
        attempt_id=ATTEMPT_ID,
        attempt=2,
        model_ref=MODEL_REF,
        data_contract_sha256="a" * 64,
        objective_config_sha256="b" * 64,
        checkpoint_format="transformer-checkpoint-v5",
        application_version="0.1.12",
        git_commit="c" * 40,
        targets=("MeanReturn",),
        milestones={
            "createdAt": "2026-08-15T20:00:00.000Z",
            "firstInputCommittedAt": "2026-08-15T20:00:01.000Z",
            "inputClosedAt": "2026-08-15T20:01:00.000Z",
            "workerCompletedAt": "2026-08-15T20:02:00.000Z",
            "publishedAt": "2026-08-15T20:02:01.000Z",
        },
        durations={
            "firstInputWaitMs": 1000.0,
            "eofWaitMs": 59000.0,
            "queueWaitMs": 25.0,
            "workerStartupMs": 450.0,
            "trainingMs": 115000.0,
            "checkpointSerializationMs": 200.0,
            "checkpointPublicationMs": 100.0,
            "modelPublicationMs": 1000.0,
            "remoteFitMs": 121000.0,
        },
        counts={
            "attempts": 2,
            "recoveries": 1,
            "inputPayloads": 4,
            "inputRows": 1000,
            "inputBytes": 9000,
        },
    )


def test_fit_run_schemas_and_index_template_are_closed():
    for name in ("run-summary.schema.json", "run-document.schema.json"):
        schema = json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
    template_path = (
        CONTRACT_ROOT / "opensearch" / "metrics-runs-v3.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert "data_stream" not in template
    assert template["index_patterns"] == ["metrics-runs-v3"]
    assert template["template"]["settings"] == {"number_of_replicas": 0}
    assert template["template"]["mappings"]["dynamic"] == "strict"


def test_run_document_is_idempotent_and_does_not_expose_storage_path():
    document = build_run_document(
        _summary(),
        deployment_id="hp800g9.home",
        byte_count=1234,
        sha256="d" * 64,
    )

    assert document["schema"] == "inventory.metrics.fit-run.v3"
    assert document["runId"] == JOB_ID
    assert document["counts"]["recoveries"] == 1
    assert "path" not in json.dumps(document)


def test_nonfinite_or_extended_summary_is_rejected():
    nonfinite = deepcopy(_summary())
    nonfinite["durations"]["trainingMs"] = math.inf
    with pytest.raises(ValueError, match="must be finite"):
        validate_run_summary(nonfinite)

    extended = deepcopy(_summary())
    extended["durations"]["unexpectedMs"] = 1.0
    with pytest.raises(ValueError, match="Additional properties"):
        validate_run_summary(extended)

def test_summary_identity_has_a_cross_language_golden_digest():
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the contract test"
    result = subprocess.run(
        [
            node,
            str(CONTRACT_ROOT / "fixtures" / "summary_id_sha256.mjs"),
            str(CONTRACT_ROOT / "fixtures" / "summary-identity.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = build_run_document(
        _summary(),
        deployment_id="hp800g9.home",
        byte_count=1234,
        sha256="d" * 64,
    )

    assert result.stdout.strip() == document["summaryId"]
