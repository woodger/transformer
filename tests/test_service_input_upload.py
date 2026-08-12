from dataclasses import replace

import pytest

from app.contracts.worker.v2.config import ModelConfig
from app.service.application.input_models import (
    CommittedInput,
    InputUploadJob,
    InputUploadMetadata,
)
from app.service.application.services.input_upload import (
    InputKindMismatch,
    InputUploadLifecycle,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, InputState


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
        model_config=ModelConfig(seq_len=2, feature_dim=3),
    )
    return replace(job, **overrides)


def _metadata(**overrides):
    metadata = InputUploadMetadata(
        job_id="job-id",
        client_execution_id="execution-id",
        fencing_token=7,
        payload_id="payload-id",
        ordinal=1,
        schema_id="inventory.sequence.fit.v2",
        input_kind="fit",
        data_contract_sha256="a" * 64,
        rows=10,
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


@pytest.mark.parametrize(
    ("metadata", "exception", "code"),
    [
        (_metadata(input_kind="predict"), InputKindMismatch, None),
        (_metadata(fencing_token=6), ServiceError, ErrorCode.STALE_FENCE),
        (
            _metadata(data_contract_sha256="b" * 64),
            ServiceError,
            ErrorCode.MODEL_SCHEMA_MISMATCH,
        ),
    ],
)
def test_upload_authorization_rejects_contract_and_fence_mismatch(
    metadata,
    exception,
    code,
):
    lifecycle = _lifecycle(UploadStore(_job()))

    with pytest.raises(exception) as error:
        lifecycle.authorize("inventory", "job-id", metadata)

    if code is not None:
        assert error.value.code is code


def test_exact_replay_uses_current_input_frontier_without_recommit():
    existing = CommittedInput(
        job_id="job-id",
        payload_id="payload-id",
        ordinal=1,
        schema_id="inventory.sequence.fit.v2",
        data_contract_sha256="a" * 64,
        rows=10,
        batches=1,
        byte_count=512,
        sha256="b" * 64,
        schema_fingerprint="c" * 64,
        relative_path="jobs/job-id/input.arrow",
        storage_class="recovery",
        source_width=6,
        feature_dim=3,
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
