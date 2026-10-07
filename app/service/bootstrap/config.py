import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TypedDict, cast

from app import config as defaults
from app.contracts.flight.v23.constants import MAX_PAYLOADS_PER_JOB
from app.project import PROJECT_ROOT

ENV_PREFIX = "TRANSFORMER_"
LEGACY_ENV_PREFIX = "TRANSFORMER_FLIGHT_"
_NON_ENVIRONMENT_FIELDS = frozenset({
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


class _FlightServiceOverrides(TypedDict, total=False):
    runtime_dir: str
    host: str
    port: int
    tls_cert_file: str | None
    tls_key_file: str | None
    tls_ca_file: str | None
    tls_require_client_cert: bool
    max_message_bytes: int
    target_batch_bytes: int
    max_batch_bytes: int
    max_payload_bytes: int
    max_rows_per_payload: int
    max_payloads_per_job: int
    max_job_bytes: int
    max_active_jobs_per_subject: int
    cpu_capacity: int
    ticket_ttl_seconds: int
    cancel_grace_seconds: float
    shutdown_drain_seconds: float
    retention_seconds: int
    maintenance_interval_seconds: int
    subprocess_timeout_seconds: float
    input_idle_timeout_seconds: float
    acquire_idle_grace_seconds: float


@dataclass(frozen=True, slots=True)
class FlightServiceConfig:
    runtime_dir: str = defaults.RUNTIME_DIR_DEFAULT
    host: str = defaults.HOST_DEFAULT
    port: int = defaults.PORT_DEFAULT
    tls_cert_file: str | None = None
    tls_key_file: str | None = None
    tls_ca_file: str | None = None
    tls_require_client_cert: bool = False

    max_message_bytes: int = defaults.MAX_MESSAGE_BYTES_DEFAULT
    target_batch_bytes: int = defaults.TARGET_BATCH_BYTES_DEFAULT
    max_batch_bytes: int = defaults.MAX_BATCH_BYTES_DEFAULT
    max_payload_bytes: int = defaults.MAX_PAYLOAD_BYTES_DEFAULT
    max_rows_per_payload: int = defaults.MAX_ROWS_PER_PAYLOAD_DEFAULT
    max_payloads_per_job: int = MAX_PAYLOADS_PER_JOB
    max_job_bytes: int = defaults.MAX_JOB_BYTES_DEFAULT
    max_active_jobs_per_subject: int = (
        defaults.MAX_ACTIVE_JOBS_PER_SUBJECT_DEFAULT
    )

    cpu_capacity: int = defaults.CPU_WORKERS
    ticket_ttl_seconds: int = defaults.TICKET_TTL_SECONDS_DEFAULT
    cancel_grace_seconds: float = defaults.CANCEL_GRACE_SECONDS_DEFAULT
    shutdown_drain_seconds: float = defaults.SHUTDOWN_DRAIN_SECONDS_DEFAULT

    retention_seconds: int = defaults.RETENTION_SECONDS
    maintenance_interval_seconds: int = (
        defaults.MAINTENANCE_INTERVAL_SECONDS_DEFAULT
    )
    subprocess_timeout_seconds: float = (
        defaults.SUBPROCESS_TIMEOUT_SECONDS_DEFAULT
    )
    input_idle_timeout_seconds: float = (
        defaults.INPUT_IDLE_TIMEOUT_SECONDS_DEFAULT
    )
    acquire_idle_grace_seconds: float = (
        defaults.ACQUIRE_IDLE_GRACE_SECONDS_DEFAULT
    )

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
    def telemetry_dir(self) -> str:
        return os.path.join(
            os.path.dirname(self.models_dir),
            "telemetry",
        )

    @property
    def lock_path(self) -> str:
        return os.path.join(self.runtime_dir, "service.lock")

    def validate(self) -> "FlightServiceConfig":
        runtime_dir: object = object.__getattribute__(self, "runtime_dir")
        if not isinstance(runtime_dir, str) or not runtime_dir:
            raise ValueError("runtime_dir must not be empty")
        host: object = object.__getattribute__(self, "host")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        port: object = object.__getattribute__(self, "port")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or port < 0
            or port > 65535
        ):
            raise ValueError("port must be between 0 and 65535")
        require_client_cert: object = object.__getattribute__(
            self,
            "tls_require_client_cert",
        )
        if not isinstance(require_client_cert, bool):
            raise ValueError("tls_require_client_cert must be a boolean")

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
            value = cast(object, getattr(self, name))
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in (
            "cancel_grace_seconds",
            "shutdown_drain_seconds",
            "subprocess_timeout_seconds",
            "input_idle_timeout_seconds",
            "acquire_idle_grace_seconds",
        ):
            value = cast(object, getattr(self, name))
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
    overrides: Mapping[str, object] | None = None,
) -> FlightServiceConfig:
    values: dict[str, object] = {}
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
    typed_values = cast(_FlightServiceOverrides, values)
    return FlightServiceConfig(**typed_values).validate()


def _parse_environment_value(
    value: str,
    annotation: object,
    key: str,
) -> str | int | float | bool:
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
