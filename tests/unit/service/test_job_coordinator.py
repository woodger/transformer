import json
from types import SimpleNamespace

from app.contracts.worker.v10.config import ModelConfig, TrainConfig
from app.contracts.worker.v10.objective import default_objective, ml_contract
from app.service.adapters.inbound.flight.constants import CREATE_ACTION
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.application.messages.jobs import (
    JobCreated,
    ServiceLimits,
)
from app.service.domain.job import ExecutionState, InputState


def test_create_dispatch_maps_public_gpu_to_internal_cuda_for_flight_v9():
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
            initialization={"kind": "random"},
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
        "version": 2,
        "profile": "research-dividend-events-v2",
        "data_contract_sha256": "a" * 64,
        "seq_len": 2,
        "feature_dim": 1,
        "target_schema_id": "inventory.target.v2",
    }
    train_config = TrainConfig()
    objective = default_objective()
    contract = ml_contract(objective)
    request = {
        "request_id": request_id,
        "idempotency_key": "create:1",
        "job_id": job_id,
        "client_execution_id": execution_id,
        "operation": "fit",
        "device": "gpu",
        "prediction_column": "out",
        "data_contract": data_contract,
        "ml_contract": contract,
        "model_label": "daily",
        "model_selector": None,
        "model_ref": None,
        "model_config": ModelConfig(seq_len=2, feature_dim=1),
        "train_config": train_config,
        "objective_config": objective,
        "initialization_kind": "random",
    }
    document = {
        "contract": "transformer-flight",
        "version": 9,
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
    assert captured[0].requested_device == "cuda"
    assert captured[0].model_config == ModelConfig(
        seq_len=2,
        feature_dim=1,
    )
    assert result["contract"] == "transformer-flight"
    assert result["version"] == 9
    assert result["jobId"] == job_id
    assert result["device"] == {"requested": "gpu", "selected": None}
    assert result["ownership"] == {
        "clientExecutionId": execution_id,
        "fencingToken": "1",
    }
    assert result["upload"] == {
        "descriptorPath": [
            "transformer",
            "v9",
            "jobs",
            job_id,
            "inputs",
            "{ordinal}",
        ],
        "schemaId": "inventory.sequence.fit.v3",
        "oneDoPutIsOneSemanticPayload": True,
    }


def test_capabilities_and_health_expose_gpu_without_cuda_backend_fields():
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
    inventory = SimpleNamespace(
        torch_version="2.12.0",
        runtime_version="13.0",
        cuda_capacity=1,
        device_count=2,
        quarantined_count=1,
    )
    service_status = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(
            device_inventory=inventory,
            cpu_capacity=2,
            limits=limits,
        ),
        health=lambda: SimpleNamespace(
            ready=True,
            draining=False,
            ledger_available=True,
            device_inventory=inventory,
            runtime_storage=SimpleNamespace(free=100),
            recovery_storage=SimpleNamespace(free=200),
            metrics={"rpc": {}, "counters": {}, "gauges": {"gpuAvailable": True}},
        ),
    )
    coordinator = JobCoordinator(
        create_job=None,
        acquire_job=None,
        close_input=None,
        cancel_job=None,
        get_status=None,
        list_inputs=None,
        list_outputs=None,
        describe_model=None,
        service_status=service_status,
        availability=None,
    )

    capabilities = coordinator.capabilities("request-capabilities")
    health = coordinator.health("request-health")

    assert capabilities["protocolVersions"] == [9]
    assert capabilities["fitInitializations"] == [
        "random",
        "publishedModel",
    ]
    assert capabilities["mlContract"]["directLossOperators"] == [
        "SmoothL1",
        "BinaryCrossEntropyWithLogits",
        "LogMSE",
    ]
    assert capabilities["devices"] == {
        "cpu": {"available": True},
        "gpu": {
            "available": True,
            "deviceCount": 2,
            "quarantinedCount": 1,
        },
    }
    assert capabilities["queue"] == {
        "cpuCapacity": 2,
        "gpuCapacity": 1,
        "singleInstance": True,
    }
    assert capabilities["features"]["deviceAwareGpu"] is True
    assert health["gpu"] == {
        "available": True,
        "deviceCount": 2,
        "quarantinedCount": 1,
    }
