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

ML_CONTRACT = {
    "targetSchemaId": "inventory.target.v1",
    "objectiveId": "transformer.objective.target-aligned.v1",
}


def test_persisted_v4_wire_result_replays_without_new_mutation():
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
        data_contract={
            "id": "inventory.learning-dataset",
            "version": 1,
            "data_contract_sha256": "a" * 64,
            "seq_len": 2,
            "feature_dim": 1,
            "target_schema_id": "inventory.target.v1",
        },
        ml_contract=ML_CONTRACT,
        limits=limits,
    )
    wire_result = present_job_created(created)

    class ReplayLedger:
        def lookup_idempotency(self, owner, action, key):
            assert (owner, action, key) == (
                "inventory",
                "transformer.v4.job.create",
                "create:1",
            )
            return {
                "request_hash": "request-hash",
                "response": wire_result,
            }

    gateway = PostgresJobLifecycle(
        ReplayLedger(),
        JobActionNames(
            create="transformer.v4.job.create",
            acquire="transformer.v4.job.acquire",
            input_close="transformer.v4.job.input.close",
            cancel="transformer.v4.job.cancel",
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
        data_contract=created.data_contract,
        ml_contract=ML_CONTRACT,
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
