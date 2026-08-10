import math
import os
import tempfile

import pytest

from app.config import (
    ALLOW_PLAINTEXT,
    CPU_WORKERS,
    HOST_DEFAULT,
    PORT_DEFAULT,
    PROJECT_NAME,
    RETENTION_SECONDS,
)
from app.flight.config import FlightServiceConfig, load_config


def test_service_defaults_come_from_app_config(tmp_path):
    state = tmp_path / "state"
    config = FlightServiceConfig(
        runtime_dir=str(state),
    ).validate()
    assert config.runtime_dir == str(state)
    assert config.host == HOST_DEFAULT
    assert config.port == PORT_DEFAULT
    assert config.allow_plaintext is ALLOW_PLAINTEXT
    assert config.tls_cert_file is None
    assert config.tls_key_file is None
    assert config.tls_ca_file is None
    assert config.tls_require_client_cert is False
    assert config.cpu_capacity == CPU_WORKERS
    assert config.retention_seconds == RETENTION_SECONDS

    with pytest.raises(ValueError, match="plaintext Flight is disabled"):
        FlightServiceConfig(
            runtime_dir=str(state),
            allow_plaintext=False,
        ).validate()


def test_explicit_plaintext_allows_non_loopback_host(tmp_path):
    FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        host="0.0.0.0",
        allow_plaintext=True,
    ).validate()


def test_tls_and_cpu_worker_count_are_independent(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("certificate")
    key.write_text("private-key")

    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "state"),
        tls_cert_file=str(cert),
        tls_key_file=str(key),
        cpu_capacity=3,
    ).validate()

    assert config.tls_enabled is True
    assert config.cpu_capacity == 3
    assert not hasattr(config, "device")


def test_remaining_environment_and_cli_precedence():
    config = load_config(
        environ={
            "TRANSFORMER_MAX_ACTIVE_JOBS_PER_SUBJECT": "4",
        },
        overrides={"host": "127.0.0.4", "port": 0},
    )

    assert config.max_active_jobs_per_subject == 4
    assert config.host == "127.0.0.4"
    assert config.port == 0
    assert config.runtime_dir == os.path.join(
        tempfile.gettempdir(),
        PROJECT_NAME,
    )


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
        "runtime_dir": str(tmp_path),
        "allow_plaintext": True,
        field: value,
    }
    with pytest.raises(ValueError):
        FlightServiceConfig(**values).validate()


def test_config_rejects_payload_count_above_protocol_limit(tmp_path):
    with pytest.raises(ValueError, match="must not exceed"):
        FlightServiceConfig(
            runtime_dir=str(tmp_path / "runtime"),
            allow_plaintext=True,
            max_payloads_per_job=100_001,
        ).validate()
