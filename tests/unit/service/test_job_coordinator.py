import json
from types import SimpleNamespace

from app.contracts.worker.v6.config import ModelConfig, TrainConfig
from app.contracts.worker.v6.objective import ml_contract
from app.service.adapters.inbound.flight.constants import CREATE_ACTION
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.application.messages.jobs import (
    JobCreated,
    ServiceLimits,
)
from app.service.domain.job import ExecutionState, InputState


def test_create_dispatch_maps_neutral_command_and_result_to_flight_v4():
    captured = []
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

    def create(command):
        captured.append(command)
        return JobCreated(
            request_id=command.request_id,
            job_id=command.job_id,
            operation=command.operation,
            revision=1,
            input_state=InputState.OPEN,
            input_revision=0,
            next_input_ordinal=0,
            execution_state=ExecutionState.WAITING_INPUT,
            client_execution_id=command.client_execution_id,
            fencing_token=1,
            requested_device=command.requested_device,
            selected_device=None,
            resolved_model_ref=None,
            data_contract=command.data_contract,
            ml_contract=command.ml_contract,
            limits=limits,
        )

    coordinator = JobCoordinator(
        create_job=SimpleNamespace(create=create),
        acquire_job=None,
        close_input=None,
        cancel_job=None,
        get_status=None,
        list_inputs=None,
        list_outputs=None,
        describe_model=None,
        service_status=None,
        availability=None,
    )
    request_id = "00000000-0000-4000-8000-000000000001"
    job_id = "00000000-0000-4000-8000-000000000002"
    execution_id = "00000000-0000-4000-8000-000000000003"
    data_contract = {
        "id": "inventory.learning-dataset",
        "version": 1,
        "data_contract_sha256": "a" * 64,
        "seq_len": 2,
        "feature_dim": 1,
        "target_schema_id": "inventory.target.v1",
    }
    train_config = TrainConfig()
    contract = ml_contract(train_config)
    request = {
        "request_id": request_id,
        "idempotency_key": "create:1",
        "job_id": job_id,
        "client_execution_id": execution_id,
        "operation": "fit",
        "device": "cpu",
        "prediction_column": "out",
        "data_contract": data_contract,
        "ml_contract": contract,
        "model_label": "daily",
        "model_selector": None,
        "model_ref": None,
        "model_config": ModelConfig(seq_len=2, feature_dim=1),
        "train_config": train_config,
    }
    document = {
        "contract": "transformer-flight",
        "version": 4,
        "requestId": request_id,
        "idempotencyKey": "create:1",
        "jobId": job_id,
    }

    result = json.loads(coordinator.dispatch(
        CREATE_ACTION,
        "inventory",
        request,
        document,
    ))

    assert captured[0].owner_subject == "inventory"
    assert captured[0].model_config == ModelConfig(
        seq_len=2,
        feature_dim=1,
    )
    assert result["contract"] == "transformer-flight"
    assert result["version"] == 4
    assert result["jobId"] == job_id
    assert result["ownership"] == {
        "clientExecutionId": execution_id,
        "fencingToken": "1",
    }
    assert result["upload"] == {
        "descriptorPath": [
            "transformer",
            "v4",
            "jobs",
            job_id,
            "inputs",
            "{ordinal}",
        ],
        "schemaId": "inventory.sequence.fit.v2",
        "oneDoPutIsOneSemanticPayload": True,
    }
