import json
import os
import uuid
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pytest

from app.contracts.worker.v10.config import TrainConfig, train_config_to_manifest
from app.contracts.worker.v10.objective import default_objective
from app.service.adapters.inbound.flight.constants import (
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CREATE_ACTION,
    ErrorCode,
)
from app.service.adapters.inbound.flight.documents import (
    encode_document,
    response_document,
)
from app.service.adapters.inbound.flight.errors import to_flight_exception
from app.service.adapters.inbound.flight.output import (
    OutputHandler as FlightOutputHandler,
    _stream_batches,
)
from app.service.adapters.inbound.flight.server import TransformerFlightServer
from app.service.bootstrap.config import FlightServiceConfig
from app.service.domain.errors import ServiceError
from app.service.domain.records import OutputRecord
from tests.support.authentication import StaticAccessTokenAuthenticator

SECURITY_TRAIN_CONFIG = TrainConfig(
    epochs=1,
)
SECURITY_OBJECTIVE = default_objective()


def _auth(token="secret"):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode("ascii"))],
        timeout=5.0,
    )


def _query_body():
    return json.dumps({
        "contract": CONTRACT_NAME,
        "version": 9,
        "requestId": str(uuid.uuid4()),
    }).encode("utf-8")


def _create_fit_document(**overrides):
    document = {
        "contract": CONTRACT_NAME,
        "version": 9,
        "requestId": str(uuid.uuid4()),
        "idempotencyKey": "security-create-1",
        "jobId": str(uuid.uuid4()),
        "clientExecutionId": str(uuid.uuid4()),
        "operation": "fit",
        "device": "cpu",
        "initialization": {"kind": "random"},
        "modelLabel": "returns.daily",
        "modelConfig": {"seqLen": 2, "hidden": 8, "nhead": 2},
        "trainingConfig": train_config_to_manifest(SECURITY_TRAIN_CONFIG),
        "targets": list(SECURITY_OBJECTIVE.targets),
        "objective": SECURITY_OBJECTIVE.objective,
        "dataContract": {
            "id": "inventory.learning-dataset",
            "version": 2,
            "profile": "research-dividend-events-v2",
            "dataContractSha256": "d" * 64,
            "seqLen": 2,
            "featureDim": 1,
            "targetSchemaId": "inventory.target.v2",
        },
    }
    document.update(overrides)
    return document


class _CountingCoordinator:
    def __init__(self):
        self.calls = []
        self.error_code = None

    def dispatch(self, action, owner, request, document):
        self.calls.append((action, owner, request, document))
        if self.error_code is not None:
            raise ServiceError(self.error_code, "safe test failure")
        return encode_document(response_document(request["request_id"], ok=True))


class _CountingUpload:
    def __init__(self):
        self.calls = []

    def handle(self, owner, descriptor, reader, writer):
        self.calls.append((owner, descriptor, reader, writer))


class _CountingOutput:
    def __init__(self):
        self.info_calls = []
        self.get_calls = []

    def get_flight_info(self, owner, descriptor):
        self.info_calls.append((owner, descriptor))
        raise AssertionError("unauthenticated GetFlightInfo reached application handler")

    def do_get(self, context, owner, ticket):
        self.get_calls.append((context, owner, ticket))
        raise AssertionError("unauthenticated DoGet reached application handler")


@pytest.fixture
def protected_server(tmp_path):
    coordinator = _CountingCoordinator()
    upload = _CountingUpload()
    output = _CountingOutput()
    server = TransformerFlightServer(
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "runtime"),
            port=0,
        ),
        coordinator,
        StaticAccessTokenAuthenticator({"secret": "inventory"}),
        upload_handler=upload,
        output_handler=output,
    )
    client = flight.FlightClient(("localhost", server.port))
    try:
        yield coordinator, upload, output, server, client
    finally:
        client.close()
        server.shutdown()


@pytest.mark.parametrize("token", [None, "wrong"])
def test_authentication_middleware_rejects_every_public_rpc_before_handler(
    protected_server,
    token,
):
    coordinator, upload, output, _, client = protected_server
    options = None if token is None else _auth(token)

    with pytest.raises(flight.FlightUnauthenticatedError):
        list(client.list_actions(options=options))
    with pytest.raises(flight.FlightUnauthenticatedError):
        list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, _query_body()),
            options=options,
        ))
    with pytest.raises(flight.FlightUnauthenticatedError):
        client.get_flight_info(
            flight.FlightDescriptor.for_path("untrusted"),
            options=options,
        )
    with pytest.raises(flight.FlightUnauthenticatedError):
        client.do_get(flight.Ticket(b"untrusted"), options=options).read_all()

    writer = None
    with pytest.raises(flight.FlightUnauthenticatedError):
        writer, _ = client.do_put(
            flight.FlightDescriptor.for_path("untrusted"),
            pa.schema([("src", pa.list_(pa.float32()))]),
            options=options,
        )
        writer.done_writing()
        writer.close()

    assert coordinator.calls == []
    assert upload.calls == []
    assert output.info_calls == []
    assert output.get_calls == []


@pytest.mark.parametrize(
    "override",
    [
        {"modelLabel": "../../outside"},
        {"modelLabel": "/srv/models/checkpoint.pth"},
        {"modelConfig": {"seqLen": 2, "argv": ["--epochs", "999"]}},
        {"trainingConfig": {"epochs": 1, "checkpointPath": "/tmp/model.pth"}},
        {"predictionColumn": "out/../../checkpoint"},
        {"cliArguments": ["--device", "cpu"]},
    ],
)
def test_paths_and_arbitrary_cli_arguments_are_rejected_before_dispatch(
    protected_server,
    override,
):
    coordinator, _, _, _, client = protected_server
    body = json.dumps(_create_fit_document(**override)).encode("utf-8")

    with pytest.raises(pa.ArrowInvalid):
        list(client.do_action(
            flight.Action(CREATE_ACTION, body),
            options=_auth(),
        ))

    assert coordinator.calls == []


def test_mtls_settings_cannot_be_silently_ignored_by_plaintext_transport(tmp_path):
    certificate_authority = tmp_path / "ca.pem"
    certificate_authority.write_text("test CA", encoding="utf-8")

    with pytest.raises(ValueError, match="required with tls_ca_file"):
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "state"),
            tls_ca_file=str(certificate_authority),
            tls_require_client_cert=True,
        ).validate()

    with pytest.raises(ValueError, match="TLS must be enabled"):
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "other-state"),
            tls_require_client_cert=True,
        ).validate()


def _write_two_batch_output(path):
    output_type = pa.list_(pa.float32(), 6)
    schema = pa.schema([
        pa.field("out", output_type, nullable=False),
    ])
    batches = [
        pa.RecordBatch.from_arrays(
            [pa.array(
                [[float(offset + index) for index in range(6)]],
                type=output_type,
            )],
            schema=schema,
        )
        for offset in (0, 10)
    ]
    with pa.OSFile(os.fspath(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
    return batches


def test_do_get_opens_output_before_return_and_survives_retention_unlink(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "output.arrow"
    expected = _write_two_batch_output(path)
    access = SimpleNamespace(
        resolve=lambda owner, token: OutputRecord(
            job_id="job-id",
            ordinal=0,
            rows=2,
            batches=2,
            byte_count=path.stat().st_size,
            sha256="0" * 64,
            schema_fingerprint="1" * 64,
            relative_path="output.arrow",
            published_at=0.0,
        )
    )
    spool = SimpleNamespace(absolute_path=lambda relative: str(path))

    class CapturedStream:
        def __init__(self, schema, generator):
            self.schema = schema
            self.generator = generator

    monkeypatch.setattr(
        "app.service.adapters.inbound.flight.output.flight.GeneratorStream",
        CapturedStream,
    )
    stream = FlightOutputHandler(access, spool).do_get(
        SimpleNamespace(is_cancelled=lambda: False),
        "inventory",
        flight.Ticket(b"opaque"),
    )

    path.unlink()
    actual = list(stream.generator)

    assert stream.schema == expected[0].schema
    assert actual == expected


def test_do_get_cancellation_closes_the_active_ipc_source(tmp_path):
    path = tmp_path / "output.arrow"
    _write_two_batch_output(path)
    source = pa.memory_map(os.fspath(path), "r")
    reader = ipc.RecordBatchFileReader(source)
    context = SimpleNamespace(cancelled=False)
    context.is_cancelled = lambda: context.cancelled
    batches = _stream_batches(context, source, reader)

    assert next(batches).num_rows == 1
    context.cancelled = True
    with pytest.raises(flight.FlightCancelledError, match="CANCELLED"):
        next(batches)

    assert source.closed is True


@pytest.mark.parametrize(
    ("code", "server_exception", "client_exception"),
    [
        (ErrorCode.INVALID_ARGUMENT, pa.ArrowInvalid, pa.ArrowInvalid),
        (
            ErrorCode.UNAUTHENTICATED,
            flight.FlightUnauthenticatedError,
            flight.FlightUnauthenticatedError,
        ),
        (
            ErrorCode.PERMISSION_DENIED,
            flight.FlightUnauthorizedError,
            flight.FlightUnauthorizedError,
        ),
        (ErrorCode.NOT_FOUND, pa.ArrowKeyError, pa.ArrowKeyError),
        # PyArrow 24 lacks these three server status constructors.  The tests
        # deliberately lock down the documented non-success fallback instead
        # of claiming that the exact normative gRPC status was emitted.
        (ErrorCode.ALREADY_EXISTS, pa.ArrowInvalid, pa.ArrowInvalid),
        (ErrorCode.FAILED_PRECONDITION, pa.ArrowInvalid, pa.ArrowInvalid),
        (
            ErrorCode.RESOURCE_EXHAUSTED,
            pa.ArrowCapacityError,
            flight.FlightServerError,
        ),
        (
            ErrorCode.CANCELLED,
            flight.FlightCancelledError,
            flight.FlightCancelledError,
        ),
        (
            ErrorCode.UNAVAILABLE,
            flight.FlightUnavailableError,
            flight.FlightUnavailableError,
        ),
        (ErrorCode.INTERNAL, flight.FlightInternalError, flight.FlightInternalError),
        (
            ErrorCode.DEVICE_LOST,
            flight.FlightInternalError,
            flight.FlightInternalError,
        ),
        (
            ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
            flight.FlightInternalError,
            flight.FlightInternalError,
        ),
        (
            ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
            flight.FlightInternalError,
            flight.FlightInternalError,
        ),
        (
            ErrorCode.RECOVERY_INPUT_UNAVAILABLE,
            flight.FlightInternalError,
            flight.FlightInternalError,
        ),
    ],
)
def test_pyarrow24_error_mapping_is_non_success_and_preserves_stable_code(
    protected_server,
    code,
    server_exception,
    client_exception,
):
    coordinator, _, _, _, client = protected_server
    mapped = to_flight_exception(ServiceError(code, "safe test failure"))
    assert isinstance(mapped, server_exception)
    assert str(mapped).startswith(f"{code.value}: safe test failure")
    coordinator.error_code = code

    with pytest.raises(client_exception, match=code.value):
        list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, _query_body()),
            options=_auth(),
        ))
