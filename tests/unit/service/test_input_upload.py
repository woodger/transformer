from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy.orm import Session

import app.service.adapters.outbound.postgres.ledger.inputs as input_ledger_module
from app.contracts.semantic.v5 import ModelContract
from app.contracts.worker.v20.config import ModelConfig
from app.service.adapters.outbound.postgres.ledger.inputs import (
    InputLedgerSlice,
)
from app.service.adapters.outbound.postgres.ledger.support import (
    LedgerSessions,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.application.messages.inputs import (
    CommittedInput,
    InputUploadJob,
    InputUploadMetadata,
)
from app.service.application.services.input_upload import (
    InputUploadLifecycle,
)
from app.service.domain.errors import ServiceError
from app.service.domain.input_manifest import manifest_sha256
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from tests.fixture_documents import semantic_fixture_document

SOURCE_ENCODING = {
    "featureBlocks": [
        {"windowRows": 1, "nativeRowWidth": 3},
    ],
}


class UploadStore:
    def __init__(self, job, existing=None):
        self.job = job
        self.existing = existing

    def get_job(self, job_id, *, owner_subject):
        assert (job_id, owner_subject) == (
            self.job.job_id,
            self.job.owner_subject,
        )
        return self.job

    def find_input(self, job_id, *, ordinal, payload_id):
        assert job_id == self.job.job_id
        return self.existing


def _job(**overrides):
    contract = ModelContract.from_document(
        semantic_fixture_document("single-regression")["modelContract"],
    )
    job = InputUploadJob(
        job_id="job-id",
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.OPEN,
        requested_device="auto",
        selected_device=None,
        client_execution_id="execution-id",
        fencing_token=7,
        input_revision=1,
        next_input_ordinal=1,
        data_contract_sha256="a" * 64,
        source_encoding=SOURCE_ENCODING,
        model_contract=contract.to_document(),
        model_config=ModelConfig(
            seq_len=2,
            feature_dim=3,
            normalization_order="postNorm",
        ),
    )
    return replace(job, **overrides)


def _metadata(**overrides):
    metadata = InputUploadMetadata(
        job_id="job-id",
        client_execution_id="execution-id",
        fencing_token=7,
        payload_id="payload-id",
        ordinal=1,
        schema_id="transformer.indexed-feature-blocks.fit.v1",
        input_kind="fit",
        data_contract_sha256="a" * 64,
        chunks=1,
        rows=10,
        native_rows=(10,),
    )
    return replace(metadata, **overrides)


def _lifecycle(store, *, cuda_available=True):
    return InputUploadLifecycle(
        store,
        max_payloads=10,
        max_job_bytes=4096,
        recovery_enabled=True,
        cuda_available=lambda: cuda_available,
    )


def test_upload_authorization_selects_device_and_durable_fit_storage():
    lifecycle = _lifecycle(UploadStore(_job()))

    authorization = lifecycle.authorize(
        "inventory",
        "job-id",
        _metadata(),
    )

    assert authorization.selected_device == "cuda"
    assert authorization.storage_class == "recovery"
    assert authorization.existing is None
    assert len(authorization.upload_token) == 48


def test_upload_authorization_resolves_contract_fields_from_the_job():
    lifecycle = _lifecycle(UploadStore(_job()))

    authorization = lifecycle.authorize(
        "inventory",
        "job-id",
        _metadata(
            schema_id="untrusted-schema",
            input_kind="predict",
            data_contract_sha256="b" * 64,
        ),
    )

    assert authorization.metadata.schema_id == (
        "transformer.indexed-feature-blocks.fit.v1"
    )
    assert authorization.metadata.input_kind == "fit"
    assert authorization.metadata.data_contract_sha256 == "a" * 64


def test_upload_authorization_rejects_a_stale_mutation_lease():
    lifecycle = _lifecycle(UploadStore(_job()))

    with pytest.raises(ServiceError) as error:
        lifecycle.authorize("inventory", "job-id", _metadata(fencing_token=6))

    assert error.value.code is ErrorCode.STALE_FENCE


def test_exact_replay_uses_current_input_frontier_without_recommit():
    existing = CommittedInput(
        job_id="job-id",
        payload_id="payload-id",
        ordinal=1,
        schema_id="transformer.indexed-feature-blocks.fit.v1",
        data_contract_sha256="a" * 64,
        chunks=1,
        rows=10,
        native_rows=(10,),
        first_range_ordinal=0,
        first_example_offset=0,
        last_range_ordinal=0,
        next_example_offset=10,
        batches=1,
        byte_count=512,
        sha256="b" * 64,
        schema_fingerprint="c" * 64,
        relative_path="jobs/job-id/input.arrow",
        storage_class="recovery",
        input_revision=2,
        next_input_ordinal=2,
        queued=True,
        frontier_advanced=True,
    )
    store = UploadStore(_job(), existing)
    lifecycle = _lifecycle(store)
    authorization = lifecycle.authorize(
        "inventory",
        "job-id",
        _metadata(),
    )
    store.job = replace(
        store.job,
        input_revision=4,
        next_input_ordinal=3,
    )

    replay = lifecycle.replay(authorization)

    assert replay.input_revision == 4
    assert replay.next_input_ordinal == 3
    assert replay.queued is False
    assert replay.frontier_advanced is False


def test_predict_inputs_wait_for_closed_manifest_before_queueing(monkeypatch):
    job_id = "00000000-0000-4000-8000-000000000001"
    execution_id = "00000000-0000-4000-8000-000000000002"
    job = SimpleNamespace(
        job_id=job_id,
        client_execution_id=execution_id,
        fencing_token=1,
        operation="predict",
        input_state=InputState.OPEN.value,
        execution_state=ExecutionState.WAITING_INPUT.value,
        data_contract_sha256="a" * 64,
        total_native_rows=[0],
        next_input_ordinal=0,
        input_revision=0,
        payload_count=0,
        total_chunks=0,
        total_rows=0,
        total_bytes=0,
        waiting_for_input=False,
        revision=1,
        source_width=1,
        feature_dim=1,
        selected_device=None,
        requested_device="cpu",
    )
    uploads = [
        SimpleNamespace(
            job_id=job_id,
            client_execution_id=execution_id,
            fencing_token=1,
            storage_class="runtime",
            ordinal=ordinal,
            payload_id=(
                f"00000000-0000-4000-8000-00000000000{ordinal + 3}"
            ),
        )
        for ordinal in range(2)
    ]
    records = []
    scalar_values = [job, None, job, None, job, None, 1]
    scalars_calls = 0

    def scalar(_statement):
        if not scalar_values:
            pytest.fail("unexpected scalar query")
        return scalar_values.pop(0)

    def scalars(_statement):
        nonlocal scalars_calls
        scalars_calls += 1
        if scalars_calls == 1:
            return [0]
        if scalars_calls == 3:
            return [1]
        if scalars_calls in (2, 4, 5):
            return records
        pytest.fail("unexpected scalar collection query")

    session = cast(
        Session,
        SimpleNamespace(
            get=lambda *_args, **_kwargs: uploads.pop(0),
            scalar=scalar,
            scalars=scalars,
            add=records.append,
            delete=lambda _record: None,
            flush=lambda: None,
        ),
    )

    @contextmanager
    def transaction():
        yield session

    database = cast(Database, SimpleNamespace(transaction=transaction))
    ledger = InputLedgerSlice(LedgerSessions(database))
    monkeypatch.setattr(
        input_ledger_module,
        "decode",
        lambda record: {"job_id": record.job_id},
    )

    for ordinal in range(2):
        receipt = ledger.commit_input(
            upload_token=f"upload-{ordinal}",
            job_id=job_id,
            client_execution_id=execution_id,
            fencing_token=1,
            relative_path=f"inputs/{ordinal}.arrow",
            schema_id="transformer.indexed-feature-blocks.predict.v1",
            data_contract_sha256="a" * 64,
            chunks=1,
            rows=2,
            native_rows=(2,),
            first_range_ordinal=ordinal,
            first_example_offset=0,
            last_range_ordinal=ordinal,
            next_example_offset=2,
            batches=1,
            byte_count=16,
            sha256="b" * 64,
            schema_fingerprint="c" * 64,
            selected_device="cpu",
            max_payloads=10,
            max_job_bytes=1024,
        )

        assert receipt["queued"] is False
        assert job.input_state == InputState.OPEN.value
        assert job.execution_state == ExecutionState.WAITING_INPUT.value

    _, repeated = ledger.close_input(
        job_id,
        client_execution_id=execution_id,
        fencing_token=1,
        expected_logical_rows=4,
        expected_manifest_sha256=manifest_sha256(records),
        select_device=lambda *_args: "cpu",
        connection=session,
    )

    assert repeated is False
    assert job.input_state == InputState.CLOSED.value
    assert job.execution_state == ExecutionState.QUEUED.value
    assert job.payload_count == 2
