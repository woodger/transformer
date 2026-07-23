from configparser import ConfigParser
from pathlib import Path

from app.database.config import load_database_config
from app.flight.config import FlightServiceConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = PROJECT_ROOT / "deploy" / "systemd"


def _load_unit() -> ConfigParser:
    parser = ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    with (SYSTEMD_DIR / "transformer.service").open(encoding="utf-8") as stream:
        parser.read_file(stream)
    return parser


def test_service_uses_target_runtime_identity_and_entrypoint():
    service = _load_unit()["Service"]

    assert service["Type"] == "exec"
    assert service["User"] == "nerv"
    assert service["Group"] == "nerv"
    assert service["WorkingDirectory"] == "/home/nerv/transformer"
    assert service["EnvironmentFile"] == "/home/nerv/transformer/.env"
    assert service["ExecStart"].split() == [
        "/usr/bin/python3.11",
        "/home/nerv/transformer/app/main.py",
        "flight",
        "serve",
    ]


def test_service_preserves_runtime_storage_and_allows_cuda():
    service = _load_unit()["Service"]

    assert service["PrivateTmp"] == "no"
    assert service["PrivateDevices"] == "no"
    assert service["ProtectSystem"] == "full"


def test_service_stop_policy_allows_application_to_drain_workers():
    service = _load_unit()["Service"]
    config = FlightServiceConfig()
    timeout = int(service["TimeoutStopSec"].removesuffix("s"))

    assert service["KillSignal"] == "SIGTERM"
    assert service["KillMode"] == "mixed"
    assert service["SendSIGKILL"] == "yes"
    assert timeout >= (
        config.shutdown_drain_seconds + config.cancel_grace_seconds + 20
    )
    assert service["Restart"] == "on-failure"


def test_service_does_not_apply_database_migrations_on_start():
    service = _load_unit()["Service"]

    assert "ExecStartPre" not in service
    assert "migrations" not in service["ExecStart"]


def test_tmpfiles_policy_creates_runtime_directory_without_age_cleanup():
    lines = [
        line.split()
        for line in (
            SYSTEMD_DIR / "transformer.tmpfiles.conf"
        ).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert lines == [
        ["d", "/tmp/transformer", "0700", "nerv", "nerv", "-"],
    ]


def test_environment_example_uses_current_configuration_contract():
    lines = (
        SYSTEMD_DIR / "transformer.env.example"
    ).read_text(encoding="utf-8").splitlines()
    environment = dict(
        line.split("=", 1)
        for line in lines
        if line and not line.startswith("#") and "=" in line
    )
    keys = set(environment)
    content = "\n".join(lines)

    assert {
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "TRANSFORMER_RUNTIME_DIR",
        "TRANSFORMER_HOST",
        "TRANSFORMER_PORT",
        "TRANSFORMER_ALLOW_PLAINTEXT",
        "TRANSFORMER_CPU_CAPACITY",
        "TRANSFORMER_CUDA_CAPACITY",
    } <= keys
    assert "POSTGRES_SSLMODE" not in content
    assert "TRANSFORMER_FLIGHT_" not in content
    assert "BEARER_TOKENS_FILE" not in content
    assert "TRANSFORMER_PROFILE" not in content

    database = load_database_config(
        environ=environment,
        env_file=SYSTEMD_DIR / "transformer.env.example",
    )
    flight = load_config(environ=environment)
    assert database.port == 5432
    assert flight.runtime_dir == "/tmp/transformer"
    assert flight.host == "127.0.0.1"
    assert flight.port == 8815
