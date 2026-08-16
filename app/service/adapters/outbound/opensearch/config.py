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
    deployment_id: str
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    ca_file: str | None = None
    connect_timeout_seconds: float = 3.0
    request_timeout_seconds: float = 15.0
    max_bulk_documents: int = 500
    max_bulk_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OPENSEARCH_ENDPOINT must be an HTTP(S) origin")
        if parsed.path not in ("", "/"):
            raise ValueError("OPENSEARCH_ENDPOINT must not contain a path")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("OPENSEARCH_ENDPOINT has an invalid port") from exc
        if _DEPLOYMENT_ID.fullmatch(self.deployment_id) is None:
            raise ValueError("OPENSEARCH_DEPLOYMENT_ID is invalid")
        security_values = (self.username, self.password, self.ca_file)
        if parsed.scheme == "http":
            if any(security_values):
                raise ValueError(
                    "HTTP OpenSearch must not configure credentials or CA"
                )
            return
        if not all(security_values):
            raise ValueError(
                "HTTPS OpenSearch requires username, password and CA file"
            )
        if self.username is not None and self.username.casefold() == "admin":
            raise ValueError("OpenSearch metrics publisher must not use admin")
        if self.ca_file is None or not os.path.isfile(self.ca_file):
            raise ValueError("OPENSEARCH_CA_FILE must name a trusted CA file")


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
    required = ("OPENSEARCH_ENDPOINT", "OPENSEARCH_DEPLOYMENT_ID")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(
            "incomplete OpenSearch metrics configuration: "
            + ", ".join(missing)
        )
    return OpenSearchMetricsConfig(
        endpoint=values["OPENSEARCH_ENDPOINT"].rstrip("/"),
        deployment_id=values["OPENSEARCH_DEPLOYMENT_ID"],
        username=values.get("OPENSEARCH_USERNAME") or None,
        password=values.get("OPENSEARCH_PASSWORD") or None,
        ca_file=values.get("OPENSEARCH_CA_FILE") or None,
    )


__all__ = ["OpenSearchMetricsConfig", "load_opensearch_metrics_config"]
