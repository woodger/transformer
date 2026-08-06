import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from app.contracts.flight.v2.constants import (
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
    PREDICTION_SCHEMA_ID,
)
from app.contracts.worker.v1 import (
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    WorkerContractError,
    encode_event,
    parse_event,
    validate_document,
)

SCHEMA_ROOT = (
    Path(__file__).parents[1]
    / "app"
    / "contracts"
    / "worker"
    / "v1"
    / "schemas"
)
JOB_ID = "00000000-0000-4000-8000-000000000001"
ATTEMPT_ID = "00000000-0000-4000-8000-000000000002"
SHA256 = "0" * 64


def _model_config() -> dict:
    return {
        "seqLen": 20,
        "hidden": 256,
        "layers": 5,
        "dropout": 0.1,
        "nhead": 8,
        "mode": "relaxed",
        "outDim": 6,
        "featureDim": 4,
    }


def _training_config() -> dict:
    return {
        "lr": 0.001,
        "batchSize": 32,
        "epochs": 20,
        "patience": 5,
        "lossStage": 1,
        "lossSchedule": "epoch",
        "stageSize": 1,
        "useAmp": False,
        "weightDecay": 0.0,
        "monitor": "loss",
        "monitorMinImprovement": 0.0,
        "saveBestCheckpoint": True,
        "seed": 42,
        "deterministic": False,
    }


def _fit_manifest() -> dict:
    return {
        "contract": "transformer-worker",
        "protocolVersion": 1,
        "jobId": JOB_ID,
        "attempt": 1,
        "attemptId": ATTEMPT_ID,
        "operation": "fit",
        "device": {"kind": "cpu"},
        "inputs": [
            {
                "schemaId": "inventory.sequence.fit.v1",
                "ordinal": 0,
                "rows": 10,
                "artifact": {
                    "path": "/srv/transformer/recovery/input.arrow",
                    "byteCount": 4096,
                    "sha256": SHA256,
                },
            }
        ],
        "workspace": {"root": "/tmp/transformer/attempt"},
        "model": {"label": "forecast", "config": _model_config()},
        "training": _training_config(),
        "recovery": {
            "configSha256": SHA256,
            "manifestSha256": "1" * 64,
        },
    }


def test_worker_v1_schemas_are_valid_draft_2020_12_documents():
    schemas = sorted(SCHEMA_ROOT.glob("*.schema.json"))

    assert {path.name for path in schemas} == {
        "arrow-manifest.schema.json",
        "capabilities.schema.json",
        "command-manifest.schema.json",
        "common.schema.json",
        "event.schema.json",
        "result-manifest.schema.json",
    }
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_worker_v1_pins_the_arrow_schemas_carried_from_flight_v2():
    assert (
        FIT_INPUT_SCHEMA_ID,
        PREDICT_INPUT_SCHEMA_ID,
        PREDICTION_OUTPUT_SCHEMA_ID,
    ) == (
        FIT_SCHEMA_ID,
        PREDICT_SCHEMA_ID,
        PREDICTION_SCHEMA_ID,
    )


def test_command_manifest_has_independent_attempt_identity_and_closed_shape():
    manifest = _fit_manifest()

    assert validate_document(manifest, "command-manifest") is manifest

    with pytest.raises(WorkerContractError, match="Additional properties"):
        validate_document({**manifest, "fencingToken": "duplicate"}, "command-manifest")


def test_cuda_command_requires_an_opaque_device_assignment():
    manifest = _fit_manifest()
    manifest["device"] = {"kind": "cuda"}

    with pytest.raises(WorkerContractError, match="opaqueId"):
        validate_document(manifest, "command-manifest")


def test_worker_event_round_trip_preserves_equality_fence_and_sequence():
    encoded = encode_event(
        job_id=JOB_ID,
        attempt=3,
        attempt_id=ATTEMPT_ID,
        sequence=1,
        event_type="ready",
        payload={"pid": 1234},
    )

    assert parse_event(encoded) == {
        "contract": "transformer-worker",
        "protocolVersion": 1,
        "jobId": JOB_ID,
        "attempt": 3,
        "attemptId": ATTEMPT_ID,
        "sequence": 1,
        "type": "ready",
        "payload": {"pid": 1234},
    }


def test_worker_event_rejects_partial_and_unbounded_lines():
    valid = encode_event(
        job_id=JOB_ID,
        attempt=1,
        attempt_id=ATTEMPT_ID,
        sequence=1,
        event_type="error",
        payload={"code": "SUBPROCESS_FAILED", "message": "worker failed"},
    )

    with pytest.raises(WorkerContractError, match="newline terminated"):
        parse_event(valid.rstrip(b"\n"))
    with pytest.raises(WorkerContractError, match="size limit"):
        parse_event(b"{" + b"x" * (1024 * 1024) + b"}\n")
