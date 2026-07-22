import json
import math

import pytest

from app.flight.config import FlightServiceConfig, load_bearer_tokens, load_config


def test_plaintext_requires_explicit_opt_in(tmp_path):
    state = tmp_path / "state"
    with pytest.raises(ValueError, match="plaintext Flight is disabled"):
        FlightServiceConfig(state_dir=str(state)).validate()

    config = FlightServiceConfig(
        state_dir=str(state),
        allow_plaintext=True,
    ).validate()
    assert config.host == "127.0.0.1"


def test_explicit_plaintext_allows_non_loopback_host(tmp_path):
    FlightServiceConfig(
        state_dir=str(tmp_path),
        host="0.0.0.0",
        allow_plaintext=True,
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


def test_environment_and_cli_precedence_and_secret_token_file(tmp_path):
    token_path = tmp_path / "tokens.json"
    token_path.write_text(json.dumps({"tokens": {"secret": "inventory-prod"}}))

    config = load_config(
        environ={
            "TRANSFORMER_STATE_DIR": str(tmp_path / "state"),
            "TRANSFORMER_CPU_CAPACITY": "4",
            "TRANSFORMER_HOST": "127.0.0.3",
            "TRANSFORMER_ALLOW_PLAINTEXT": "true",
            "TRANSFORMER_BEARER_TOKENS_FILE": str(token_path),
        },
        overrides={"host": "127.0.0.4", "port": 0},
    )

    assert config.cpu_capacity == 4
    assert config.host == "127.0.0.4"
    assert config.port == 0
    assert load_bearer_tokens(config) == {"secret": "inventory-prod"}


def test_config_rejects_unknown_override():
    with pytest.raises(ValueError, match="unknown Flight configuration"):
        load_config(environ={}, overrides={"unknown": True})


def test_config_rejects_legacy_environment_namespace():
    with pytest.raises(ValueError, match="TRANSFORMER_FLIGHT_HOST"):
        load_config(
            environ={"TRANSFORMER_FLIGHT_HOST": "127.0.0.1"},
        )


def test_config_rejects_legacy_bind_host_environment_variable():
    with pytest.raises(ValueError, match="TRANSFORMER_BIND_HOST"):
        load_config(
            environ={"TRANSFORMER_BIND_HOST": "127.0.0.1"},
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
        "allow_plaintext": True,
        field: value,
    }
    with pytest.raises(ValueError):
        FlightServiceConfig(**values).validate()


def test_config_rejects_payload_count_that_cannot_fit_seal_document(tmp_path):
    with pytest.raises(ValueError, match="seal manifest"):
        FlightServiceConfig(
            state_dir=str(tmp_path),
            allow_plaintext=True,
            max_payloads_per_job=401,
        ).validate()
