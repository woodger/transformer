from __future__ import annotations

from dataclasses import dataclass

from app.service.domain.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class GetModelTopologyQuery:
    owner_subject: str
    request_id: str
    model_ref: str


@dataclass(frozen=True, slots=True)
class ModelTopologyResult:
    request_id: str
    model_ref: str
    model_definition_sha256: str
    nodes: tuple[JsonObject, ...]
    edges: tuple[JsonObject, ...]


__all__ = ["GetModelTopologyQuery", "ModelTopologyResult"]
