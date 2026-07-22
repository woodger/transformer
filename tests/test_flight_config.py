import json
import math

import pytest

from app.flight.config import FlightServiceConfig, load_bearer_tokens, load_config


def test_plaintext_requires_explicit_development_or_lan_profile(tmp_path):
    state = tmp_path / "state"
    with pytest.raises(ValueError, match="plaintext Flight is disabled"):
        FlightServiceConfig(state_dir=str(state)).validate()
    with pytest.raises(ValueError, match="development or lan"):
        FlightServiceConfig(
            state_dir=str(state),
            allow_plaintext=True,
            profile="production",
        ).validate()

    config = FlightServiceConfig(
        state_dir=str(state),
        allow_plaintext=True,
        profile="development",
    ).validate()
    assert config.host == "127.0.0.1"


def test_non_loopback_plaintext_requires_lan_profile(tmp_path):
    with pytest.raises(ValueError, match="explicit lan"):
        FlightServiceConfig(
            state_dir=str(tmp_path),
            host="0.0.0.0",
            allow_plaintext=True,
            profile="development",
        ).validate()

    FlightServiceConfig(
        state_dir=str(tmp_path),
        host="0.0.0.0",
        allow_plaintext=True,
        profile="lan",
    ).validate()


def test_tls_and_device_capacity_are_independent(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("certificate")
    key.write_text("private-key")

    config = FlightServiceConfig(
        state_dir=str(tmp_path / "state"),
        tls_cert_file=str(cert),
        tls_key_file=str(key),
        cpu_capacity=3,
        cuda_capacity=1,
    ).validate()

    assert config.tls_enabled is True
    assert config.cpu_capacity == 3
    assert not hasattr(config, "device")


def test_config_precedence_and_secret_token_file(tmp_path):
    config_path = tmp_path / "flight.json"
    token_path = tmp_path / "tokens.json"
    token_path.write_text(json.dumps({"tokens": {"secret": "inventory-prod"}}))
    config_path.write_text(json.dumps({
        "stateDir": str(tmp_path / "from-file"),
        "host": "127.0.0.2",
        "profile": "development",
        "allowPlaintext": True,
        "bearerTokensFile": str(token_path),
        "cpuCapacity": 1,
    }))

    config = load_config(
        str(config_path),
        environ={
            "TRANSFORMER_FLIGHT_CPU_CAPACITY": "4",
            "TRANSFORMER_FLIGHT_HOST": "127.0.0.3",
        },
        overrides={"port": 0},
    )

    assert config.cpu_capacity == 4
    assert config.host == "127.0.0.3"
    assert config.port == 0
    assert load_bearer_tokens(config) == {"secret": "inventory-prod"}


@pytest.mark.parametrize("field", ("unknown", "bindHost", "bind_host"))
def test_config_rejects_unknown_field(tmp_path, field):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({field: True}))
    with pytest.raises(ValueError, match="unknown Flight configuration"):
        load_config(str(path), environ={})


def test_config_rejects_legacy_host_environment_variable():
    with pytest.raises(ValueError, match="TRANSFORMER_FLIGHT_BIND_HOST"):
        load_config(
            environ={"TRANSFORMER_FLIGHT_BIND_HOST": "127.0.0.1"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_payloads_per_job", True),
        ("max_payload_bytes", math.nan),
        ("max_rows_per_payload", 1.5),
        ("ticket_ttl_seconds", True),
        ("cancel_grace_seconds", math.inf),
    ],
)
def test_config_rejects_boolean_fractional_and_nonfinite_quotas(
    tmp_path,
    field,
    value,
):
    values = {
        "state_dir": str(tmp_path),
        "profile": "development",
        "allow_plaintext": True,
        field: value,
    }
    with pytest.raises(ValueError):
        FlightServiceConfig(**values).validate()


def test_config_rejects_payload_count_that_cannot_fit_seal_document(tmp_path):
    with pytest.raises(ValueError, match="seal manifest"):
        FlightServiceConfig(
            state_dir=str(tmp_path),
            profile="development",
            allow_plaintext=True,
            max_payloads_per_job=401,
        ).validate()
