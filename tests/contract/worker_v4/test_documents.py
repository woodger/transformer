from __future__ import annotations

import io
import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from app.contracts.flight.v4.constants import (
    FIT_SCHEMA_ID,
    PREDICT_SCHEMA_ID,
    PREDICTION_SCHEMA_ID,
)
from app.contracts.worker.v4 import (
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    WorkerContractError,
    encode_control_message,
    encode_event,
    parse_control_message,
    parse_event,
    validate_document,
)
from app.contracts.worker.v4.config import TrainConfig, train_config_to_manifest
from app.contracts.worker.v4.objective import ml_contract
from app.project import PROJECT_ROOT
from app.worker.application.inputs import DurableInputStream

SCHEMA_ROOT = (
    PROJECT_ROOT / "app"
    / "contracts"
    / "worker"
    / "v4"
    / "schemas"
)
JOB_ID = "00000000-0000-4000-8000-000000000001"
ATTEMPT_ID = "00000000-0000-4000-8000-000000000002"
PAYLOAD_0 = "00000000-0000-4000-8000-000000000003"
PAYLOAD_1 = "00000000-0000-4000-8000-000000000004"
SHA256 = "0" * 64
DATA_CONTRACT_SHA256 = "1" * 64


def _model_config() -> dict:
    return {
        "seqLen": 2,
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
        "nhead": 2,
        "mode": "relaxed",
        "outDim": 6,
        "featureDim": 2,
    }


def _training_config() -> dict:
    return train_config_to_manifest(TrainConfig(batch_size=4, epochs=2))


def _data_contract() -> dict:
    return {
        "id": "inventory.learning-dataset",
        "version": 1,
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "seqLen": 2,
        "featureDim": 2,
        "targetSchemaId": "inventory.target.v1",
    }


def _input(ordinal: int, *, revision: int, rows: int, byte_count: int) -> dict:
    return {
        "schemaId": FIT_INPUT_SCHEMA_ID,
        "ordinal": ordinal,
        "commitRevision": revision,
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "rows": rows,
        "artifact": {
            "path": f"/srv/transformer/recovery/{ordinal}.arrow",
            "byteCount": byte_count,
            "sha256": f"{ordinal + 2:064x}",
        },
    }


def _fit_manifest(*, closed: bool = False) -> dict:
    return {
        "contract": "transformer-worker",
        "protocolVersion": 4,
        "jobId": JOB_ID,
        "attempt": 1,
        "attemptId": ATTEMPT_ID,
        "operation": "fit",
        "device": {"kind": "cpu"},
        "inputs": [_input(0, revision=1, rows=2, byte_count=10)],
        "inputRevision": 1,
        "inputClosed": closed,
        "manifestSha256": SHA256 if closed else None,
        "workspace": {"root": "/tmp/transformer/attempt"},
        "model": {"label": "forecast", "config": _model_config()},
        "training": _training_config(),
        "dataContract": _data_contract(),
        "mlContract": ml_contract(TrainConfig(batch_size=4, epochs=2)),
        "recovery": {
            "configSha256": SHA256,
            "dataContractSha256": DATA_CONTRACT_SHA256,
            "objectiveConfigSha256": ml_contract(
                TrainConfig(batch_size=4, epochs=2)
            )["objectiveConfigSha256"],
            "manifestSha256": SHA256 if closed else None,
        },
    }


class _Emitter:
    def __init__(self):
        self.events = []

    def input_waiting(self, **payload):
        self.events.append(("waiting", payload))

    def input_ack(self, **payload):
        self.events.append(("ack", payload))


def test_worker_v4_schemas_are_valid_draft_2020_12_documents():
    schemas = sorted(SCHEMA_ROOT.glob("*.schema.json"))
    assert {path.name for path in schemas} == {
        "arrow-manifest.schema.json",
        "capabilities.schema.json",
        "command-manifest.schema.json",
        "common.schema.json",
        "control-message.schema.json",
        "event.schema.json",
        "result-manifest.schema.json",
        "training-metrics.schema.json",
    }
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_worker_v4_pins_flight_v4_arrow_schema_ids():
    assert (
        FIT_INPUT_SCHEMA_ID,
        PREDICT_INPUT_SCHEMA_ID,
        PREDICTION_OUTPUT_SCHEMA_ID,
    ) == (FIT_SCHEMA_ID, PREDICT_SCHEMA_ID, PREDICTION_SCHEMA_ID)


def test_command_manifest_separates_internal_attempt_from_external_fence():
    manifest = _fit_manifest()
    assert validate_document(manifest, "command-manifest") is manifest

    for forbidden in ("attemptFence", "fencingToken", "clientExecutionId"):
        invalid = {**manifest, forbidden: "external-ownership"}
        with pytest.raises(WorkerContractError, match="Additional properties"):
            validate_document(invalid, "command-manifest")


def test_command_manifest_requires_explicit_open_or_closed_input_identity():
    open_manifest = _fit_manifest()
    validate_document(open_manifest, "command-manifest")

    closed = _fit_manifest(closed=True)
    validate_document(closed, "command-manifest")

    missing_digest = deepcopy(closed)
    missing_digest["manifestSha256"] = None
    with pytest.raises(WorkerContractError, match="not of type 'string'"):
        validate_document(missing_digest, "command-manifest")


def test_worker_event_and_control_envelopes_are_equality_fenced_and_sequenced():
    event = encode_event(
        job_id=JOB_ID,
        attempt=3,
        attempt_id=ATTEMPT_ID,
        sequence=1,
        event_type="ready",
        payload={"pid": 1234, "nextOrdinal": 0, "inputRevision": 0},
    )
    assert parse_event(event)["attemptId"] == ATTEMPT_ID

    control = encode_control_message(
        job_id=JOB_ID,
        attempt=3,
        attempt_id=ATTEMPT_ID,
        sequence=1,
        message_type="input.committed",
        payload={
            "inputRevision": 2,
            "input": _input(1, revision=2, rows=3, byte_count=20),
        },
    )
    assert parse_control_message(control)["payload"]["input"]["ordinal"] == 1

    for partial in (event.rstrip(b"\n"), control.rstrip(b"\n")):
        parser = parse_event if partial.startswith(b'{"attempt"') else parse_control_message
        with pytest.raises(WorkerContractError, match="newline terminated"):
            parser(partial)


def test_durable_input_stream_blocks_at_frontier_then_accepts_eof():
    committed = encode_control_message(
        job_id=JOB_ID,
        attempt=1,
        attempt_id=ATTEMPT_ID,
        sequence=1,
        message_type="input.committed",
        payload={
            "inputRevision": 2,
            "input": _input(1, revision=2, rows=3, byte_count=20),
        },
    )
    closed = encode_control_message(
        job_id=JOB_ID,
        attempt=1,
        attempt_id=ATTEMPT_ID,
        sequence=2,
        message_type="input.closed",
        payload={
            "inputRevision": 2,
            "payloadCount": 2,
            "totalRows": 5,
            "totalBytes": 30,
            "manifestSha256": SHA256,
        },
    )
    emitter = _Emitter()
    stream = DurableInputStream(
        _fit_manifest(),
        io.BytesIO(committed + closed),
        emitter,
    )

    assert [item["ordinal"] for item in stream.items()] == [0, 1]
    assert stream.closed is True
    assert stream.input_revision == 2
    assert stream.manifest_sha256 == SHA256
    assert emitter.events == [
        ("waiting", {"next_ordinal": 1, "input_revision": 1}),
        (
            "ack",
            {"ordinal": 1, "next_ordinal": 2, "input_revision": 2},
        ),
        ("waiting", {"next_ordinal": 2, "input_revision": 2}),
    ]


def test_durable_input_stream_replays_exact_duplicate_controls_safely():
    duplicate = encode_control_message(
        job_id=JOB_ID,
        attempt=1,
        attempt_id=ATTEMPT_ID,
        sequence=1,
        message_type="input.committed",
        payload={
            "inputRevision": 1,
            "input": _input(0, revision=1, rows=2, byte_count=10),
        },
    )
    committed = encode_control_message(
        job_id=JOB_ID,
        attempt=1,
        attempt_id=ATTEMPT_ID,
        sequence=2,
        message_type="input.committed",
        payload={
            "inputRevision": 2,
            "input": _input(1, revision=2, rows=3, byte_count=20),
        },
    )
    closed = encode_control_message(
        job_id=JOB_ID,
        attempt=1,
        attempt_id=ATTEMPT_ID,
        sequence=3,
        message_type="input.closed",
        payload={
            "inputRevision": 2,
            "payloadCount": 2,
            "totalRows": 5,
            "totalBytes": 30,
            "manifestSha256": SHA256,
        },
    )
    emitter = _Emitter()
    stream = DurableInputStream(
        _fit_manifest(),
        io.BytesIO(duplicate + committed + closed),
        emitter,
    )

    assert [item["ordinal"] for item in stream.items()] == [0, 1]
    assert [payload["ordinal"] for event, payload in emitter.events if event == "ack"] == [
        0,
        1,
    ]


def test_durable_input_stream_rejects_gaps_wrong_identity_and_bad_eof_totals():
    cases = [
        encode_control_message(
            job_id=JOB_ID,
            attempt=1,
            attempt_id=ATTEMPT_ID,
            sequence=1,
            message_type="input.committed",
            payload={
                "inputRevision": 3,
                "input": _input(2, revision=3, rows=1, byte_count=5),
            },
        ),
        encode_control_message(
            job_id=JOB_ID,
            attempt=2,
            attempt_id=ATTEMPT_ID,
            sequence=1,
            message_type="input.closed",
            payload={
                "inputRevision": 1,
                "payloadCount": 1,
                "totalRows": 2,
                "totalBytes": 10,
                "manifestSha256": SHA256,
            },
        ),
        encode_control_message(
            job_id=JOB_ID,
            attempt=1,
            attempt_id=ATTEMPT_ID,
            sequence=1,
            message_type="input.closed",
            payload={
                "inputRevision": 1,
                "payloadCount": 1,
                "totalRows": 99,
                "totalBytes": 10,
                "manifestSha256": SHA256,
            },
        ),
    ]
    messages = ("not contiguous", "identity differs", "row count differs")
    for control, message in zip(cases, messages, strict=True):
        stream = DurableInputStream(
            _fit_manifest(),
            io.BytesIO(control),
            _Emitter(),
        )
        with pytest.raises(WorkerContractError, match=message):
            list(stream.items())
