import hashlib
import json
import os
from datetime import datetime, UTC
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace
import uuid

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pytest

from app.database.models import OutputTicket
from app.flight.auth import BearerAuthMiddlewareFactory
from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    ACTIONS,
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CREATE_ACTION,
    ErrorCode,
)
from app.flight.contract import encode_document, response_document, validate_action_request
from app.flight.coordinator import JobCoordinator
from app.flight.errors import ServiceError, to_flight_exception
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.output import OutputHandler, _stream_batches
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool


def _auth(token="secret"):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode("ascii"))],
        timeout=5.0,
    )


def _query_body():
    return json.dumps({
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": str(uuid.uuid4()),
    }).encode("utf-8")


def _create_fit_document(**overrides):
    document = {
        "contract": CONTRACT_NAME,
        "version": 1,
        "requestId": str(uuid.uuid4()),
        "idempotencyKey": "security-create-1",
        "operation": "fit",
        "device": "cpu",
        "modelLabel": "returns.daily",
        "modelConfig": {"seqLen": 2, "hidden": 8, "nhead": 2},
        "trainingConfig": {"epochs": 1},
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
            allow_plaintext=True,
        ),
        coordinator,
        {"secret": "inventory"},
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


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        ({"has whitespace": "inventory"}, "printable ASCII"),
        ({"café": "inventory"}, "printable ASCII"),
        ({"secret": "inventory\nadmin"}, "control characters"),
        ({"x" * 4097: "inventory"}, "printable ASCII"),
    ],
)
def test_bearer_credentials_reject_unsafe_header_and_subject_values(tokens, message):
    with pytest.raises(ValueError, match=message):
        BearerAuthMiddlewareFactory(
            tokens,
            OperationalMetrics(),
            JsonLogger("transformer.flight.security-test"),
        )


def test_mtls_settings_cannot_be_silently_ignored_by_plaintext_transport(tmp_path):
    certificate_authority = tmp_path / "ca.pem"
    certificate_authority.write_text("test CA", encoding="utf-8")

    with pytest.raises(ValueError, match="required with tls_ca_file"):
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "state"),
            allow_plaintext=True,
            tls_ca_file=str(certificate_authority),
            tls_require_client_cert=True,
        ).validate()

    with pytest.raises(ValueError, match="TLS must be enabled"):
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "other-state"),
            allow_plaintext=True,
            tls_require_client_cert=True,
        ).validate()


def _write_two_batch_output(path):
    schema = pa.schema([("out", pa.list_(pa.float32()))])
    batches = [
        pa.record_batch({
            "out": pa.array(
                [[float(offset + index) for index in range(6)]],
                type=pa.list_(pa.float32()),
            )
        })
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
    ledger = SimpleNamespace(
        resolve_ticket=lambda token, owner_subject: {"relative_path": "output.arrow"}
    )
    spool = SimpleNamespace(absolute_path=lambda relative: str(path))

    class CapturedStream:
        def __init__(self, schema, generator):
            self.schema = schema
            self.generator = generator

    monkeypatch.setattr("app.flight.output.flight.GeneratorStream", CapturedStream)
    stream = OutputHandler(SimpleNamespace(), ledger, spool).do_get(
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


@pytest.fixture
def published_output_server(tmp_path, postgres_ledger):
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        port=0,
        allow_plaintext=True,
        disk_min_free_bytes=1,
    ).validate()
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    ledger = postgres_ledger
    job_id = str(uuid.uuid4())
    ledger.create_job(
        job_id=job_id,
        owner_subject="inventory",
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        input_model_ref="mdl_seed",
        model_config={
            "seq_len": 2,
            "hidden": 8,
            "layers": 1,
            "dropout": 0.0,
            "nhead": 2,
            "context_mode": "relaxed",
            "out_dim": 6,
            "feature_dim": 1,
        },
    )
    ledger.seal_job(job_id, manifest_hash="a" * 64, manifest=[])
    ledger.queue_job(job_id, selected_device="cpu")
    running = ledger.claim_next_job("cpu")
    path = spool.attempt_output_path(job_id, running["attempt"], 0)
    spool.ensure_parent(path)
    batches = _write_two_batch_output(path)
    raw = Path(path).read_bytes()
    output = {
        "ordinal": 0,
        "rows": 2,
        "batches": 2,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_fingerprint": hashlib.sha256(
            batches[0].schema.serialize().to_pybytes()
        ).hexdigest(),
        "relative_path": spool.relative_path(path),
    }
    ledger.publish_outputs(job_id, running["attempt"], [output], result={})
    coordinator = JobCoordinator(
        config,
        ledger,
        spool,
        cuda_available=lambda: False,
    )
    server = TransformerFlightServer(
        config,
        coordinator,
        {"secret": "inventory", "other": "other-subject"},
        output_handler=OutputHandler(config, ledger, spool),
    )
    client = flight.FlightClient(("localhost", server.port))
    try:
        yield ledger, job_id, client
    finally:
        client.close()
        server.shutdown()


def test_output_descriptor_and_ticket_are_owner_bound_and_expired_ticket_fails(
    published_output_server,
):
    ledger, job_id, client = published_output_server
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job_id, "outputs", "0"
    )

    with pytest.raises(pa.ArrowKeyError):
        client.get_flight_info(descriptor, options=_auth("other"))

    info = client.get_flight_info(descriptor, options=_auth())
    ticket = info.endpoints[0].ticket
    assert b"/" not in ticket.ticket
    assert job_id.encode("ascii") not in ticket.ticket
    with pytest.raises(flight.FlightUnauthorizedError, match="PERMISSION_DENIED"):
        client.do_get(ticket, options=_auth("other")).read_all()

    with ledger.transaction() as connection:
        record = connection.get(
            OutputTicket,
            hashlib.sha256(ticket.ticket).hexdigest(),
        )
        record.expires_at = datetime.fromtimestamp(0, UTC)
    with pytest.raises(pa.ArrowInvalid, match=r"FAILED_PRECONDITION.*expired"):
        client.do_get(ticket, options=_auth()).read_all()

    with pytest.raises(pa.ArrowKeyError, match="NOT_FOUND"):
        client.do_get(flight.Ticket(b"x" * 257), options=_auth()).read_all()


def test_client_cancellation_reaches_active_do_get_and_closes_output(
    published_output_server,
    monkeypatch,
):
    _, job_id, client = published_output_server
    import app.flight.output as output_module

    original = output_module._stream_batches
    resume = threading.Event()
    closed = threading.Event()
    opened_sources = []

    def observable_stream(context, source, reader):
        opened_sources.append(source)
        batches = original(context, source, reader)
        try:
            yield next(batches)
            assert resume.wait(5), "test did not release the active DoGet"
            yield from batches
        finally:
            batches.close()
            closed.set()

    monkeypatch.setattr(output_module, "_stream_batches", observable_stream)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job_id, "outputs", "0"
    )
    ticket = client.get_flight_info(descriptor, options=_auth()).endpoints[0].ticket
    reader = client.do_get(ticket, options=_auth())
    assert reader.read_chunk().data.num_rows == 1

    reader.cancel()
    resume.set()

    assert closed.wait(5)
    assert len(opened_sources) == 1
    assert opened_sources[0].closed is True


def _generate_tls_certificate(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for the runtime TLS integration test")
    certificate = tmp_path / "server-cert.pem"
    private_key = tmp_path / "server-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            os.fspath(private_key),
            "-out",
            os.fspath(certificate),
            "-sha256",
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return certificate, private_key


def _generate_mtls_certificates(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for the runtime mTLS integration test")
    ca_certificate = tmp_path / "ca-cert.pem"
    ca_private_key = tmp_path / "ca-key.pem"
    server_certificate = tmp_path / "mtls-server-cert.pem"
    server_private_key = tmp_path / "mtls-server-key.pem"
    server_request = tmp_path / "mtls-server.csr"
    client_certificate = tmp_path / "client-cert.pem"
    client_private_key = tmp_path / "client-key.pem"
    client_request = tmp_path / "client.csr"
    commands = (
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", os.fspath(ca_private_key),
            "-out", os.fspath(ca_certificate),
            "-sha256", "-days", "1", "-nodes", "-subj", "/CN=Flight-Test-CA",
        ],
        [
            openssl, "req", "-newkey", "rsa:2048",
            "-keyout", os.fspath(server_private_key),
            "-out", os.fspath(server_request),
            "-nodes", "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        [
            openssl, "x509", "-req", "-in", os.fspath(server_request),
            "-CA", os.fspath(ca_certificate),
            "-CAkey", os.fspath(ca_private_key),
            "-CAcreateserial", "-out", os.fspath(server_certificate),
            "-days", "1", "-sha256", "-copy_extensions", "copy",
        ],
        [
            openssl, "req", "-newkey", "rsa:2048",
            "-keyout", os.fspath(client_private_key),
            "-out", os.fspath(client_request),
            "-nodes", "-subj", "/CN=inventory",
        ],
        [
            openssl, "x509", "-req", "-in", os.fspath(client_request),
            "-CA", os.fspath(ca_certificate),
            "-CAkey", os.fspath(ca_private_key),
            "-CAcreateserial", "-out", os.fspath(client_certificate),
            "-days", "1", "-sha256",
        ],
    )
    for command in commands:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=30,
        )
    return (
        ca_certificate,
        server_certificate,
        server_private_key,
        client_certificate,
        client_private_key,
    )


def test_real_tls_server_requires_bearer_authentication(tmp_path):
    certificate, private_key = _generate_tls_certificate(tmp_path)
    coordinator = _CountingCoordinator()
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        tls_cert_file=str(certificate),
        tls_key_file=str(private_key),
    ).validate()
    server = TransformerFlightServer(config, coordinator, {"secret": "inventory"})
    client = flight.FlightClient(
        flight.Location.for_grpc_tls("localhost", server.port),
        tls_root_certs=certificate.read_bytes(),
    )
    try:
        assert tuple(item.type for item in client.list_actions(options=_auth())) == ACTIONS
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(client.list_actions())
    finally:
        client.close()
        server.shutdown()


def test_real_mtls_server_rejects_missing_client_certificate(tmp_path):
    (
        ca_certificate,
        server_certificate,
        server_private_key,
        client_certificate,
        client_private_key,
    ) = _generate_mtls_certificates(tmp_path)
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        port=0,
        tls_cert_file=str(server_certificate),
        tls_key_file=str(server_private_key),
        tls_ca_file=str(ca_certificate),
        tls_require_client_cert=True,
    ).validate()
    server = TransformerFlightServer(
        config,
        _CountingCoordinator(),
        {"secret": "inventory"},
    )
    location = flight.Location.for_grpc_tls("localhost", server.port)
    client_without_certificate = None
    authenticated_client = None
    try:
        client_without_certificate = flight.FlightClient(
            location,
            tls_root_certs=ca_certificate.read_bytes(),
        )
        with pytest.raises(flight.FlightUnavailableError):
            list(client_without_certificate.list_actions(options=_auth()))

        authenticated_client = flight.FlightClient(
            location,
            tls_root_certs=ca_certificate.read_bytes(),
            cert_chain=client_certificate.read_bytes(),
            private_key=client_private_key.read_bytes(),
        )
        assert tuple(
            item.type
            for item in authenticated_client.list_actions(options=_auth())
        ) == ACTIONS
    finally:
        if authenticated_client is not None:
            authenticated_client.close()
        if client_without_certificate is not None:
            client_without_certificate.close()
        server.shutdown()


def test_tls_and_plaintext_transports_do_not_change_explicit_cuda_policy(
    tmp_path,
    postgres_ledger,
):
    certificate, private_key = _generate_tls_certificate(tmp_path)
    configs = (
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "plain-state"),
            allow_plaintext=True,
            disk_min_free_bytes=1,
        ).validate(),
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "tls-state"),
            tls_cert_file=str(certificate),
            tls_key_file=str(private_key),
            disk_min_free_bytes=1,
        ).validate(),
    )

    for index, config in enumerate(configs):
        spool = Spool(config.runtime_dir, tmp_path / f"models-{index}").initialize()
        ledger = postgres_ledger
        coordinator = JobCoordinator(
            config,
            ledger,
            spool,
            cuda_available=lambda: False,
        )
        document = _create_fit_document(
            idempotencyKey=f"cuda-policy-{index}",
            device="cuda",
        )
        request = validate_action_request(CREATE_ACTION, document)

        with pytest.raises(ServiceError) as error:
            coordinator.create("inventory", request, document)

        assert error.value.code == ErrorCode.DEVICE_UNAVAILABLE
        assert ledger.list_jobs() == []


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
