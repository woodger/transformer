from configparser import ConfigParser
from pathlib import Path
from tempfile import gettempdir

from app.config import (
    ALLOW_PLAINTEXT,
    CPU_WORKERS,
    HOST_DEFAULT,
    PORT_DEFAULT,
    PROJECT_NAME,
)
from app.database.config import load_database_config
from app.flight.config import FlightServiceConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DOCUMENT = PROJECT_ROOT / "docs" / "deployment" / "systemd.md"
README = PROJECT_ROOT / "readme.md"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def _configuration_block(heading: str, language: str) -> str:
    document = SYSTEMD_DOCUMENT.read_text(encoding="utf-8")
    section = document.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return section.split(f"```{language}\n", 1)[1].split("\n```", 1)[0]


def _load_unit() -> ConfigParser:
    parser = ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(_configuration_block("Создать unit-файл", "ini"))
    return parser


def test_service_uses_target_runtime_identity_and_entrypoint():
    unit = _load_unit()
    service = unit["Service"]

    assert (
        unit["Unit"]["Documentation"]
        == "file:/opt/transformer/docs/deployment/systemd.md"
    )
    assert service["Type"] == "exec"
    assert service["User"] == "nerv"
    assert service["Group"] == "nerv"
    assert service["WorkingDirectory"] == "/opt/transformer"
    assert unit["Unit"]["ConditionPathExists"] == "/opt/transformer/app/main.py"
    assert (
        unit["Unit"]["ConditionPathIsExecutable"]
        == "/opt/transformer/.venv/bin/python"
    )
    assert unit["Unit"]["StartLimitIntervalSec"] == "60s"
    assert unit["Unit"]["StartLimitBurst"] == "5"
    assert "EnvironmentFile" not in service
    assert service["ExecStart"].split() == [
        "/opt/transformer/.venv/bin/python",
        "/opt/transformer/app/main.py",
        "flight",
        "serve",
        "--host=0.0.0.0",
        "--port=8815",
        "--allow-plaintext",
    ]


def test_deployment_has_one_python_environment_contract():
    document = SYSTEMD_DOCUMENT.read_text(encoding="utf-8")

    assert document.count(
        "/usr/bin/python3 -m venv --clear /opt/transformer/.venv"
    ) == 1
    assert "/usr/bin/python3." not in document
    assert "ExecStart=/home/nerv/transformer" not in document
    assert "EnvironmentFile=" not in _configuration_block(
        "Создать unit-файл",
        "ini",
    )
    assert "requirements.txt" in document


def test_readme_quick_start_creates_a_clean_project_environment():
    document = README.read_text(encoding="utf-8")
    section = document.split("## Быстрый старт\n", 1)[1].split("\n## ", 1)[0]

    assert "/usr/bin/python3 -m venv --clear .venv" in section
    assert ".venv/bin/python -m pip install -r requirements.txt" in section


def test_initial_deployment_uses_a_clean_tracked_release_and_preserves_runtime_data():
    document = SYSTEMD_DOCUMENT.read_text(encoding="utf-8")

    assert 'test "$(id -un)" = nerv' in document
    assert 'test -z "$(git status --porcelain)"' in document
    assert "git ls-files --error-unmatch requirements.txt" in document
    assert "git -C /home/nerv/transformer archive --format=tar HEAD" in document
    assert "test ! -e /opt/transformer" in document
    assert "sudo systemctl disable --now transformer" in document
    assert "install -m 0600 /home/nerv/transformer/.env /opt/transformer/.env" in document
    assert "for runtime_directory in models recovery" in document
    assert "`models/` и `recovery/`" in document
    assert "tar -C /opt/transformer -xf -" in document
    assert "-C /home/nerv/transformer -cf - ." not in document


def test_deployment_preflight_precedes_unit_enablement():
    document = SYSTEMD_DOCUMENT.read_text(encoding="utf-8")

    assert document.index("## Проверить runtime до unit") < document.index(
        "## Создать unit-файл"
    )
    assert document.index("## Применить миграции") < document.index(
        "## Создать unit-файл"
    )
    assert document.index("## Создать unit-файл") < document.index(
        "## Запустить сервис"
    )


def test_service_stop_policy_allows_application_to_drain_workers():
    service = _load_unit()["Service"]
    config = FlightServiceConfig()
    timeout = int(service["TimeoutStopSec"].removesuffix("s"))

    assert service["KillMode"] == "mixed"
    assert service["RestartSec"] == "5s"
    assert timeout >= (
        config.shutdown_drain_seconds + config.cancel_grace_seconds + 20
    )
    assert service["Restart"] == "on-failure"


def test_service_does_not_apply_database_migrations_on_start():
    service = _load_unit()["Service"]

    assert "ExecStartPre" not in service
    assert "migrations" not in service["ExecStart"]


def test_requirement_manifest_is_exact_and_complete():
    requirements = [
        line
        for line in REQUIREMENTS.read_text(
            encoding="utf-8",
        ).splitlines()
        if line and not line.startswith("#")
    ]
    package_names = {
        line.split("==", 1)[0].lower()
        for line in requirements
    }

    assert all(line.count("==") == 1 for line in requirements)
    assert {
        "alembic",
        "jsonschema",
        "numpy",
        "psycopg",
        "psycopg-binary",
        "pyarrow",
        "pytest",
        "python-dotenv",
        "ruff",
        "sqlalchemy",
        "torch",
    } <= package_names


def test_environment_example_uses_current_configuration_contract():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    environment = dict(
        line.split("=", 1)
        for line in lines
        if line and not line.startswith("#") and "=" in line
    )
    keys = set(environment)
    content = "\n".join(lines)

    assert keys == {
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    }
    assert "POSTGRES_SSLMODE" not in content
    assert "TRANSFORMER_FLIGHT_" not in content
    assert "BEARER_TOKENS_FILE" not in content
    assert "TRANSFORMER_PROFILE" not in content
    assert "TRANSFORMER_HOST" not in content
    assert "TRANSFORMER_PORT" not in content

    database = load_database_config(
        environ=environment,
        env_file=ENV_EXAMPLE,
    )
    flight = load_config(environ=environment)
    assert database.port == 5432
    assert flight.runtime_dir == str(Path(gettempdir()) / PROJECT_NAME)
    assert flight.host == HOST_DEFAULT
    assert flight.port == PORT_DEFAULT
    assert flight.allow_plaintext is ALLOW_PLAINTEXT
    assert flight.cpu_capacity == CPU_WORKERS
