from app.contracts.flight.v12.constants import (
    ACQUIRE_ACTION,
    CANCEL_ACTION,
    CREATE_ACTION,
    INPUT_CLOSE_ACTION,
)
from app.service.adapters.inbound.flight.presentation import present_job_created
from app.service.adapters.outbound.postgres.job_lifecycle import (
    JobActionNames,
    PostgresJobLifecycle,
)
from app.service.application.messages.jobs import (
    CreateJobCommand,
    JobCreated,
    ServiceLimits,
)
from app.service.domain.job import ExecutionState, InputState
from tests.support.consumer_neutral import model_contract

MODEL_CONTRACT = model_contract(
    "single-regression",
    seq_len=2,
    feature_dim=1,
)
MODEL_CONTRACT_DOCUMENT = MODEL_CONTRACT.to_document()
SEMANTIC_DIGESTS = MODEL_CONTRACT.digests("a" * 64)


def test_persisted_create_result_replays_without_new_mutation():
    limits = ServiceLimits(
        max_message_bytes=1024,
        target_batch_bytes=512,
        max_batch_bytes=1024,
        max_payload_bytes=2048,
        max_rows_per_payload=100,
        max_payloads_per_job=10,
        max_job_bytes=4096,
        max_active_jobs_per_subject=2,
        max_page_items=100,
        input_idle_timeout_seconds=30.0,
    )
    created = JobCreated(
        request_id="request-id",
        job_id="job-id",
        operation="fit",
        revision=1,
        input_state=InputState.OPEN,
        input_revision=0,
        next_input_ordinal=0,
        execution_state=ExecutionState.WAITING_INPUT,
        client_execution_id="execution-id",
        fencing_token=1,
        requested_device="cpu",
        selected_device=None,
        resolved_model_ref=None,
        source_encoding={
            "kind": "indexedFeatureBlocks",
            "featureBlocks": [
                {"position": 0, "windowRows": 1, "nativeRowWidth": 1},
            ],
        },
        data_contract={
            "identity": "test.dataset",
            "revision": 1,
            "profile": "test.profile",
            "dataContractSha256": "a" * 64,
            "seqLen": 2,
            "featureDim": 1,
        },
        model_contract=MODEL_CONTRACT_DOCUMENT,
        semantic_digests=SEMANTIC_DIGESTS,
        job_config_sha256="b" * 64,
        limits=limits,
        initialization={"kind": "random"},
    )
    wire_result = present_job_created(created)
    stored_result = {
        "result_type": "job_created",
        "request_id": created.request_id,
        "job_id": created.job_id,
        "operation": created.operation,
        "revision": created.revision,
        "input_state": created.input_state.value,
        "input_revision": created.input_revision,
        "next_input_ordinal": created.next_input_ordinal,
        "execution_state": created.execution_state.value,
        "client_execution_id": created.client_execution_id,
        "fencing_token": created.fencing_token,
        "requested_device": created.requested_device,
        "selected_device": created.selected_device,
        "resolved_model_ref": created.resolved_model_ref,
        "source_encoding": created.source_encoding,
        "data_contract": created.data_contract,
        "model_contract": created.model_contract,
        "semantic_digests": created.semantic_digests,
        "job_config_sha256": created.job_config_sha256,
        "initialization": created.initialization,
        "limits": {
            "max_message_bytes": limits.max_message_bytes,
            "target_batch_bytes": limits.target_batch_bytes,
            "max_batch_bytes": limits.max_batch_bytes,
            "max_payload_bytes": limits.max_payload_bytes,
            "max_rows_per_payload": limits.max_rows_per_payload,
            "max_payloads_per_job": limits.max_payloads_per_job,
            "max_job_bytes": limits.max_job_bytes,
            "max_active_jobs_per_subject": limits.max_active_jobs_per_subject,
            "max_page_items": limits.max_page_items,
            "input_idle_timeout_seconds": limits.input_idle_timeout_seconds,
        },
    }

    class ReplayLedger:
        def lookup_idempotency(self, owner, action, key):
            assert (owner, action, key) == (
                "inventory",
                CREATE_ACTION,
                "create:1",
            )
            return {
                "request_hash": "request-hash",
                "response": stored_result,
            }

    gateway = PostgresJobLifecycle(
        ReplayLedger(),
        JobActionNames(
            create=CREATE_ACTION,
            acquire=ACQUIRE_ACTION,
            input_close=INPUT_CLOSE_ACTION,
            cancel=CANCEL_ACTION,
        ),
    )
    command = CreateJobCommand(
        owner_subject="inventory",
        request_id="request-id",
        idempotency_key="create:1",
        request_hash="request-hash",
        job_id="job-id",
        client_execution_id="execution-id",
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        source_encoding=created.source_encoding,
        data_contract=created.data_contract,
        model_contract=MODEL_CONTRACT_DOCUMENT,
        semantic_digests=SEMANTIC_DIGESTS,
        initialization_kind="random",
    )

    def fail_prepare(_model):
        raise AssertionError("replay must not prepare a new job")

    def fail_preflight():
        raise AssertionError("replay must not check new-job availability")

    outcome = gateway.create(
        command,
        max_active_jobs=2,
        preflight=fail_preflight,
        prepare=fail_prepare,
    )

    assert outcome.replayed is True
    assert present_job_created(outcome.result) == wire_result
