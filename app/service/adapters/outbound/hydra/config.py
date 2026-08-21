from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from app.project import PROJECT_ROOT

HYDRA_ENDPOINT_ENVIRONMENT_VARIABLE = "HYDRA_ENDPOINT"
INTROSPECTION_PATH = "/admin/oauth2/introspect"
REQUIRED_AUDIENCE = "transformer"
REQUIRED_SCOPE = "transformer:invoke"
INTROSPECTION_TIMEOUT_SECONDS = 3.0

_FORBIDDEN_CONFIGURATION_KEYS = (
    "ORY_HYDRA_AUDIENCE",
    "ORY_HYDRA_SCOPE",
    "ORY_HYDRA_INTROSPECTION_TIMEOUT",
    "ORY_HYDRA_INTROSPECTION_TIMEOUT_SECONDS",
)


@dataclass(frozen=True, slots=True)
class HydraAuthorizationCacheConfig:
    positive_ttl_seconds: float = 15.0
    negative_ttl_seconds: float = 2.0
    max_entries: int = 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("positive_ttl_seconds", self.positive_ttl_seconds),
            ("negative_ttl_seconds", self.negative_ttl_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Hydra authorization cache {name} is invalid")
        if isinstance(self.max_entries, bool) or self.max_entries <= 0:
            raise ValueError(
                "Hydra authorization cache max_entries is invalid"
            )


@dataclass(frozen=True, slots=True)
class HydraIntrospectionConfig:
    endpoint: str
    authorization_cache: HydraAuthorizationCacheConfig = field(
        default_factory=HydraAuthorizationCacheConfig
    )

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                f"{HYDRA_ENDPOINT_ENVIRONMENT_VARIABLE} must be an HTTP(S) "
                "Hydra Admin API base URL"
            )
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(
                f"{HYDRA_ENDPOINT_ENVIRONMENT_VARIABLE} has an invalid port"
            ) from exc


def load_hydra_introspection_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | os.PathLike[str] | None = None,
) -> HydraIntrospectionConfig:
    environment = os.environ if environ is None else environ
    if env_file is None:
        env_file = Path(PROJECT_ROOT) / ".env"
    file_values = {
        key: value
        for key, value in dotenv_values(env_file).items()
        if isinstance(value, str)
    }
    values = {**file_values, **dict(environment)}
    forbidden = [
        key for key in _FORBIDDEN_CONFIGURATION_KEYS if values.get(key)
    ]
    if forbidden:
        raise ValueError(
            "Hydra audience, scope and timeout are fixed by Transformer: "
            + ", ".join(forbidden)
        )
    endpoint = values.get(HYDRA_ENDPOINT_ENVIRONMENT_VARIABLE)
    if not endpoint:
        raise ValueError(
            f"{HYDRA_ENDPOINT_ENVIRONMENT_VARIABLE} is required"
        )
    return HydraIntrospectionConfig(endpoint=endpoint.rstrip("/"))


__all__ = [
    "HYDRA_ENDPOINT_ENVIRONMENT_VARIABLE",
    "INTROSPECTION_PATH",
    "INTROSPECTION_TIMEOUT_SECONDS",
    "REQUIRED_AUDIENCE",
    "REQUIRED_SCOPE",
    "HydraAuthorizationCacheConfig",
    "HydraIntrospectionConfig",
    "load_hydra_introspection_config",
]
