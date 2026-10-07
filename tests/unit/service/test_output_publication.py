import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from app.contracts.flight.v23.arrow import canonical_prediction_schema
from app.contracts.semantic.v6 import ModelContract
from app.contracts.worker.v21 import PREDICTION_OUTPUT_SCHEMA_ID
from app.contracts.worker.v21.config import ModelConfig
from app.contracts.worker.v21.model_definition import resolved_semantic_digests
from app.service.adapters.observability import OperationalMetrics
from app.service.adapters.outbound.artifacts.publication import WorkerArtifactPublisher
from app.service.adapters.outbound.artifacts.spool import Spool
from app.service.application.ports.workers import ExecutionInput
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import ExecutionJobRecord
from tests.fixture_documents import semantic_fixture_document


@pytest.fixture
def publication(tmp_path):
    spool = Spool(tmp_path / "runtime", tmp_path / "models").initialize()
    contract = ModelContract.from_document(
        semantic_fixture_document("single-regression")["modelContract"],
    )
    config = ModelConfig.from_tuning(contract.model_tuning, seq_len=1, feature_dim=1)
    digests = resolved_semantic_digests(contract, "a" * 64, config)
    job = ExecutionJobRecord(
        job_id="11111111-1111-4111-8111-111111111111",
        owner_subject="inventory",
        operation="predict",
        input_state=InputState.CLOSED,
        execution_state=ExecutionState.RUNNING,
        input_revision=2,
        selected_device="cpu",
        model_label=None,
        input_model_ref="mdl_published",
        prediction_column="prediction",
        source_encoding={"featureBlocks": [{"windowRows": 1, "nativeRowWidth": 1}]},
        model_config=config,
        training_config=None,
        data_contract={"dataContractSha256": "a" * 64, "seqLen": 1, "featureDim": 1},
        model_contract=contract.to_document(),
        semantic_digests=digests,
        config_hash="b" * 64,
        manifest_sha256="c" * 64,
        feature_dim=1,
        input_frame_count=2,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
        attempt_id="22222222-2222-4222-8222-222222222222",
    )
    inputs = tuple(ExecutionInput(
        ordinal=ordinal,
        commit_revision=ordinal + 1,
        schema_id="transformer.indexed-feature-blocks.predict.v1",
        data_contract_sha256="a" * 64,
        chunks=1,
        rows=1,
        native_rows=(1,),
        first_range_ordinal=0,
        first_example_offset=ordinal,
        last_range_ordinal=0,
        next_example_offset=ordinal + 1,
        batches=1,
        byte_count=100,
        sha256="d" * 64,
        absolute_path=spool.input_path(job.job_id, ordinal),
        storage_class="runtime",
    ) for ordinal in range(2))
    artifacts = []
    for item in inputs:
        path = Path(spool.attempt_output_path(job.job_id, job.attempt, item.ordinal))
        # Каталоги и файлы создаёт Worker; service должен синхронизировать их сам.
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = canonical_prediction_schema(job.prediction_column, contract.target_contract)
        column = pa.FixedSizeListArray.from_arrays(pa.array([0.25], type=pa.float32()), 1)
        with pa.OSFile(str(path), "wb") as sink:
            with ipc.new_file(sink, schema) as writer:
                writer.write_table(pa.Table.from_arrays([column], schema=schema))

        artifacts.append({
            "schemaId": PREDICTION_OUTPUT_SCHEMA_ID,
            "ordinal": item.ordinal,
            "commitRevision": item.commit_revision,
            "dataContractSha256": item.data_contract_sha256,
            "modelDefinitionSha256": digests["modelDefinitionSha256"],
            "rows": item.rows,
            "artifact": {
                "path": str(path),
                "byteCount": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        })
    published = []

    def publish_outputs(_job_id, _attempt, outputs, *, attempt_id, result):
        published.extend(outputs)
        return result

    publisher = WorkerArtifactPublisher(
        SimpleNamespace(publish_outputs=publish_outputs),
        spool,
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        metrics=OperationalMetrics(),
    )
    return publisher, spool, job, inputs, {"artifacts": artifacts}, published


def test_prediction_publication_references_synchronized_files_and_directories(
    publication,
    monkeypatch,
):
    publisher, spool, job, inputs, result, published = publication
    synchronized = set()
    real_fsync = os.fsync

    def observe_fsync(descriptor):
        real_fsync(descriptor)
        synchronized.add(os.readlink(f"/proc/self/fd/{descriptor}"))

    monkeypatch.setattr(os, "fsync", observe_fsync)
    publisher.publish_outputs_from_manifest(job, inputs, result)

    assert [output["ordinal"] for output in published] == [0, 1]
    for output in published:
        path = Path(spool.absolute_path(output["relative_path"]))
        assert str(path) in synchronized
        assert all(
            str(parent) in synchronized
            for parent in path.parents
            if parent.is_relative_to(spool.runtime_dir)
        )
        assert output["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with pa.memory_map(str(path), "r") as source:
            assert ipc.RecordBatchFileReader(source).read_all().to_pydict() == {
                "prediction": [[0.25]],
            }


@pytest.mark.parametrize("failure", ["file-0", "file-1", "outputs", "attempt", "runtime"])
def test_prediction_publication_fsync_failure_leaves_all_outputs_unpublished(
    publication,
    monkeypatch,
    failure,
):
    publisher, spool, job, inputs, result, published = publication
    first = Path(spool.attempt_output_path(job.job_id, job.attempt, 0))
    failing_path = {
        "file-0": str(first),
        "file-1": spool.attempt_output_path(job.job_id, job.attempt, 1),
        "outputs": str(first.parent),
        "attempt": str(first.parent.parent),
        "runtime": spool.runtime_dir,
    }[failure]
    real_fsync = os.fsync

    def fail_fsync(descriptor):
        if os.readlink(f"/proc/self/fd/{descriptor}") == failing_path:
            raise OSError("injected output fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected output fsync failure"):
        publisher.publish_outputs_from_manifest(job, inputs, result)

    assert published == []
    with pa.memory_map(str(first), "r") as source:
        assert ipc.RecordBatchFileReader(source).read_all().to_pydict() == {
            "prediction": [[0.25]],
        }
