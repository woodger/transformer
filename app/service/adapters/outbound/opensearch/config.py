from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from app.project import PROJECT_ROOT

_KEYS = (
    "OPENSEARCH_ENDPOINT",
    "OPENSEARCH_USERNAME",
    "OPENSEARCH_PASSWORD",
    "OPENSEARCH_CA_FILE",
    "OPENSEARCH_DEPLOYMENT_ID",
)
_DEPLOYMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class OpenSearchMetricsConfig:
    endpoint: str
    username: str
    password: str = field(repr=False)
    ca_file: str
    deployment_id: str
    connect_timeout_seconds: float = 3.0
    request_timeout_seconds: float = 15.0
    max_bulk_documents: int = 500
    max_bulk_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OPENSEARCH_ENDPOINT must be an HTTPS origin")
        if parsed.path not in ("", "/"):
            raise ValueError("OPENSEARCH_ENDPOINT must not contain a path")
        if not self.username or not self.password:
            raise ValueError("OpenSearch credentials must not be empty")
        if self.username.casefold() == "admin":
            raise ValueError("OpenSearch metrics publisher must not use admin")
        if not os.path.isfile(self.ca_file):
            raise ValueError("OPENSEARCH_CA_FILE must name a trusted CA file")
        if _DEPLOYMENT_ID.fullmatch(self.deployment_id) is None:
            raise ValueError("OPENSEARCH_DEPLOYMENT_ID is invalid")


def load_opensearch_metrics_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | os.PathLike[str] | None = None,
) -> OpenSearchMetricsConfig | None:
    environment = os.environ if environ is None else environ
    if env_file is None:
        env_file = Path(PROJECT_ROOT) / ".env"
    file_values = {
        key: value
        for key, value in dotenv_values(env_file).items()
        if isinstance(value, str)
    }
    values = {**file_values, **dict(environment)}
    configured = [key for key in _KEYS if values.get(key)]
    if not configured:
        return None
    missing = [key for key in _KEYS if not values.get(key)]
    if missing:
        raise ValueError(
            "incomplete OpenSearch metrics configuration: "
            + ", ".join(missing)
        )
    return OpenSearchMetricsConfig(
        endpoint=values["OPENSEARCH_ENDPOINT"].rstrip("/"),
        username=values["OPENSEARCH_USERNAME"],
        password=values["OPENSEARCH_PASSWORD"],
        ca_file=values["OPENSEARCH_CA_FILE"],
        deployment_id=values["OPENSEARCH_DEPLOYMENT_ID"],
    )


__all__ = ["OpenSearchMetricsConfig", "load_opensearch_metrics_config"]
