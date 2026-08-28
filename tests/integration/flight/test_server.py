import json
import uuid

import pyarrow.flight as flight
import pytest

from app.service.adapters.inbound.flight.constants import (
    ACTIONS,
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
)
from app.service.adapters.inbound.flight.documents import (
    encode_document,
    response_document,
)
from app.service.adapters.inbound.flight.server import TransformerFlightServer
from app.service.bootstrap.config import FlightServiceConfig
from tests.support.authentication import StaticAccessTokenAuthenticator


class StubCoordinator:
    def __init__(self):
        self.calls = []

    def dispatch(self, action, owner, request, document):
        self.calls.append((action, owner, request, document))
        return encode_document(response_document(
            request["request_id"],
            subject=owner,
        ))


class FailingCoordinator(StubCoordinator):
    def dispatch(self, action, owner, request, document):
        raise RuntimeError("secret postgresql://private-control-plane")


def call_options(token="secret"):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode())],
        timeout=5.0,
    )


def action_body(**overrides):
    document = {
        "contract": CONTRACT_NAME,
        "version": 6,
        "requestId": str(uuid.uuid4()),
    }
    document.update(overrides)
    return json.dumps(document).encode()


@pytest.fixture
def control_server(tmp_path):
    coordinator = StubCoordinator()
    server = TransformerFlightServer(
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "runtime"),
            port=0,
        ),
        coordinator,
        StaticAccessTokenAuthenticator({"secret": "inventory"}),
    )
    client = flight.FlightClient(("localhost", server.port))
    try:
        yield server, coordinator, client
    finally:
        client.close()
        server.shutdown()


def test_list_actions_advertises_exact_v6_contract(control_server):
    _, _, client = control_server

    actions = list(client.list_actions(options=call_options()))

    assert tuple(item.type for item in actions) == ACTIONS


def test_action_is_authenticated_and_validated_before_dispatch(control_server):
    _, coordinator, client = control_server
    request = action_body()

    result = list(client.do_action(
        flight.Action(CAPABILITIES_ACTION, request),
        options=call_options(),
    ))

    assert json.loads(result[0].body.to_pybytes())["subject"] == "inventory"
    assert len(coordinator.calls) == 1


def test_invalid_version_and_action_are_transport_errors(control_server):
    _, coordinator, client = control_server

    with pytest.raises(Exception, match="version must be 6"):
        list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, action_body(version=1)),
            options=call_options(),
        ))
    with pytest.raises(Exception, match="unsupported action"):
        list(client.do_action(
            flight.Action("transformer.v5.capabilities", action_body()),
            options=call_options(),
        ))

    assert coordinator.calls == []


def test_unexpected_handler_error_is_sanitized(control_server):
    server, _, client = control_server
    server.coordinator = FailingCoordinator()

    with pytest.raises(flight.FlightInternalError) as error:
        list(client.do_action(
            flight.Action(CAPABILITIES_ACTION, action_body()),
            options=call_options(),
        ))

    assert "INTERNAL: internal service error" in str(error.value)
    assert "secret" not in str(error.value)
    assert "private-control-plane" not in str(error.value)
