from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from app.project import PROJECT_ROOT

INTROSPECTION_ENVIRONMENT_VARIABLE = "ORY_HYDRA_INTROSPECTION_ENDPOINT"
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
class HydraIntrospectionConfig:
    endpoint: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != "/admin/oauth2/introspect"
        ):
            raise ValueError(
                f"{INTROSPECTION_ENVIRONMENT_VARIABLE} must be an HTTP(S) "
                "URL ending in /admin/oauth2/introspect"
            )
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(
                f"{INTROSPECTION_ENVIRONMENT_VARIABLE} has an invalid port"
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
    endpoint = values.get(INTROSPECTION_ENVIRONMENT_VARIABLE)
    if not endpoint:
        raise ValueError(
            f"{INTROSPECTION_ENVIRONMENT_VARIABLE} is required"
        )
    return HydraIntrospectionConfig(endpoint=endpoint.rstrip("/"))


__all__ = [
    "INTROSPECTION_ENVIRONMENT_VARIABLE",
    "INTROSPECTION_TIMEOUT_SECONDS",
    "REQUIRED_AUDIENCE",
    "REQUIRED_SCOPE",
    "HydraIntrospectionConfig",
    "load_hydra_introspection_config",
]
