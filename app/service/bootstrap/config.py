import math
import os
import tempfile
from dataclasses import dataclass, fields

from app.config import (
    ALLOW_PLAINTEXT,
    CPU_WORKERS,
    HOST_DEFAULT,
    PORT_DEFAULT,
    PROJECT_NAME,
    PROJECT_ROOT,
    RETENTION_SECONDS,
)
from app.contracts.flight.v3.constants import MAX_PAYLOADS_PER_JOB

ENV_PREFIX = "TRANSFORMER_"
LEGACY_ENV_PREFIX = "TRANSFORMER_FLIGHT_"
_NON_ENVIRONMENT_FIELDS = frozenset({
    "allow_plaintext",
    "cpu_capacity",
    "host",
    "port",
    "retention_seconds",
    "runtime_dir",
    "tls_ca_file",
    "tls_cert_file",
    "tls_key_file",
    "tls_require_client_cert",
})


@dataclass(frozen=True)
class FlightServiceConfig:
    runtime_dir: str = os.path.join(tempfile.gettempdir(), PROJECT_NAME)
    host: str = HOST_DEFAULT
    port: int = PORT_DEFAULT
    allow_plaintext: bool = ALLOW_PLAINTEXT

    tls_cert_file: str | None = None
    tls_key_file: str | None = None
    tls_ca_file: str | None = None
    tls_require_client_cert: bool = False

    max_message_bytes: int = 16 * 1024 * 1024
    target_batch_bytes: int = 8 * 1024 * 1024
    max_batch_bytes: int = 16 * 1024 * 1024
    max_payload_bytes: int = 512 * 1024 * 1024
    max_rows_per_payload: int = 2_000_000
    max_payloads_per_job: int = MAX_PAYLOADS_PER_JOB
    max_job_bytes: int = 64 * 1024 * 1024 * 1024
    max_active_jobs_per_subject: int = 32

    cpu_capacity: int = CPU_WORKERS
    ticket_ttl_seconds: int = 600
    cancel_grace_seconds: float = 10.0
    shutdown_drain_seconds: float = 30.0

    retention_seconds: int = RETENTION_SECONDS
    maintenance_interval_seconds: int = 60
    subprocess_timeout_seconds: float = 24 * 60 * 60
    input_idle_timeout_seconds: float = 15 * 60
    acquire_idle_grace_seconds: float = 30.0

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert_file or self.tls_key_file)

    @property
    def spool_dir(self) -> str:
        return os.path.join(self.runtime_dir, "spool")

    @property
    def models_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, "models")

    @property
    def recovery_dir(self) -> str:
        return os.path.join(
            os.path.dirname(self.models_dir),
            "recovery",
        )

    @property
    def lock_path(self) -> str:
        return os.path.join(self.runtime_dir, "service.lock")

    def validate(self) -> "FlightServiceConfig":
        if not isinstance(self.runtime_dir, str) or not self.runtime_dir:
            raise ValueError("runtime_dir must not be empty")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or self.port < 0
            or self.port > 65535
        ):
            raise ValueError("port must be between 0 and 65535")
        for name in ("allow_plaintext", "tls_require_client_cert"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

        cert_set = bool(self.tls_cert_file)
        key_set = bool(self.tls_key_file)
        if cert_set != key_set:
            raise ValueError("tls_cert_file and tls_key_file must be configured together")
        if self.tls_ca_file and not self.tls_enabled:
            raise ValueError("tls_cert_file and tls_key_file are required with tls_ca_file")
        if self.tls_require_client_cert and not self.tls_enabled:
            raise ValueError("TLS must be enabled when mTLS is required")
        if self.tls_require_client_cert and not self.tls_ca_file:
            raise ValueError("tls_ca_file is required when mTLS is enabled")
        if not self.tls_enabled:
            if not self.allow_plaintext:
                raise ValueError(
                    "plaintext Flight is disabled; configure TLS or explicitly "
                    "enable plaintext"
                )

        positive_integers = (
            "max_message_bytes",
            "target_batch_bytes",
            "max_batch_bytes",
            "max_payload_bytes",
            "max_rows_per_payload",
            "max_payloads_per_job",
            "max_job_bytes",
            "max_active_jobs_per_subject",
            "cpu_capacity",
            "ticket_ttl_seconds",
            "retention_seconds",
            "maintenance_interval_seconds",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in (
            "cancel_grace_seconds",
            "shutdown_drain_seconds",
            "subprocess_timeout_seconds",
            "input_idle_timeout_seconds",
            "acquire_idle_grace_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if self.target_batch_bytes > self.max_batch_bytes:
            raise ValueError("target_batch_bytes must not exceed max_batch_bytes")
        if self.max_batch_bytes > self.max_message_bytes:
            raise ValueError("max_batch_bytes must not exceed max_message_bytes")
        if self.max_message_bytes > self.max_payload_bytes:
            raise ValueError("max_message_bytes must not exceed max_payload_bytes")
        if self.max_payloads_per_job > MAX_PAYLOADS_PER_JOB:
            raise ValueError(
                "max_payloads_per_job must not exceed "
                f"{MAX_PAYLOADS_PER_JOB}"
            )

        for path_name in (
            "tls_cert_file",
            "tls_key_file",
            "tls_ca_file",
        ):
            path = getattr(self, path_name)
            if path is not None and not isinstance(path, str):
                raise ValueError(f"{path_name} must be a filesystem path")
            if path and not os.path.isfile(path):
                raise ValueError(f"{path_name} does not exist or is not a file")
        return self


def load_config(
    *,
    environ: dict[str, str] | None = None,
    overrides: dict | None = None,
) -> FlightServiceConfig:
    values = {}
    env = os.environ if environ is None else environ
    legacy_keys = sorted(
        key for key in env if key.startswith(LEGACY_ENV_PREFIX)
    )
    if legacy_keys:
        raise ValueError(
            "unsupported legacy Transformer environment variable(s): "
            f"{', '.join(legacy_keys)}; use the current app/config.py and "
            "TRANSFORMER_* settings"
        )
    legacy_host_key = ENV_PREFIX + "BIND_HOST"
    if legacy_host_key in env:
        raise ValueError(
            f"unknown Flight environment variable: {legacy_host_key}; "
            "configure HOST_DEFAULT in app/config.py or use --host"
        )
    for field in fields(FlightServiceConfig):
        if field.name in _NON_ENVIRONMENT_FIELDS:
            continue
        key = ENV_PREFIX + field.name.upper()
        if key not in env:
            continue
        values[field.name] = _parse_environment_value(
            env[key],
            field.type,
            key,
        )

    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})

    allowed = {field.name for field in fields(FlightServiceConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown Flight configuration field(s): {', '.join(unknown)}")
    return FlightServiceConfig(**values).validate()


def _parse_environment_value(value: str, annotation, key: str):
    annotation_text = str(annotation)
    if annotation is bool or annotation_text == "bool":
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} must be a boolean")
    if annotation is int or annotation_text == "int":
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
    if annotation is float or annotation_text == "float":
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be a number") from exc
    return value
