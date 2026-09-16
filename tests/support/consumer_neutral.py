from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v3 import ModelContract
from app.contracts.worker.v14.model_config import ModelConfig
from app.contracts.worker.v14.model_definition import resolved_semantic_digests

SEMANTIC_FIXTURES = (
    Path(__file__).parents[2]
    / "app"
    / "contracts"
    / "semantic"
    / "v3"
    / "fixtures"
)
DATA_CONTRACT_SHA256 = "d" * 64

_GEOMETRY_BY_CONTRACT: dict[int, tuple[int, int]] = {}


def model_contract(
    fixture: str = "multi-target-shared-resource",
    *,
    seq_len: int | None = None,
    feature_dim: int | None = None,
    hidden: int | None = None,
    layers: int | None = None,
    dropout: float | None = None,
    nhead: int | None = None,
    mode: str | None = None,
) -> ModelContract:
    document = fixture_document(fixture)
    raw_contract = document["modelContract"]
    if not isinstance(raw_contract, dict):
        raise AssertionError("semantic fixture has no modelContract")
    contract = cast(JsonObject, deepcopy(raw_contract))
    raw_tuning = contract["modelTuning"]
    if not isinstance(raw_tuning, dict):
        raise AssertionError("semantic fixture has no modelTuning")
    tuning = cast(JsonObject, raw_tuning)
    overrides: tuple[tuple[str, object | None], ...] = (
        ("hiddenWidth", hidden),
        ("encoderLayerCount", layers),
        ("dropoutProbability", dropout),
        ("attentionHeadCount", nhead),
        ("missingValuePolicy", mode),
    )
    for field, value in overrides:
        if value is not None:
            tuning[field] = value
    result = ModelContract.from_document(contract)
    fixture_geometry = _fixture_geometry(document)
    if seq_len is not None or feature_dim is not None:
        # Geometry is deliberately not part of ModelContract. Callers that
        # need a non-fixture geometry use it through data_contract().
        _GEOMETRY_BY_CONTRACT[id(result)] = (
            seq_len if seq_len is not None else fixture_geometry[0],
            feature_dim if feature_dim is not None else fixture_geometry[1],
        )
    else:
        _GEOMETRY_BY_CONTRACT[id(result)] = fixture_geometry
    return result


def data_contract(contract: ModelContract) -> JsonObject:
    geometry = tensor_geometry(contract)
    return {
        "dataContractSha256": DATA_CONTRACT_SHA256,
        **geometry,
    }


def tensor_geometry(contract: ModelContract) -> JsonObject:
    seq_len, feature_dim = _GEOMETRY_BY_CONTRACT.get(
        id(contract),
        _fixture_geometry(fixture_document("multi-target-shared-resource")),
    )
    return {"seqLen": seq_len, "featureDim": feature_dim}


def semantic_digests(
    contract: ModelContract,
    data_contract_sha256: str,
) -> JsonObject:
    geometry = tensor_geometry(contract)
    model_config = ModelConfig.from_tuning(
        contract.model_tuning,
        seq_len=cast(int, geometry["seqLen"]),
        feature_dim=cast(int, geometry["featureDim"]),
    )
    return resolved_semantic_digests(
        contract,
        data_contract_sha256,
        model_config,
    )


def fixture_document(name: str) -> JsonObject:
    path = SEMANTIC_FIXTURES / f"{name}.json"
    with path.open(encoding="utf-8") as source:
        value: object = json.load(source)
    if not isinstance(value, dict):
        raise AssertionError("semantic fixture must be an object")
    return cast(JsonObject, value)


def _fixture_geometry(document: JsonObject) -> tuple[int, int]:
    geometry = document.get("tensorGeometry")
    if not isinstance(geometry, dict):
        raise AssertionError("semantic fixture has no tensorGeometry")
    seq_len = geometry.get("seqLen")
    feature_dim = geometry.get("featureDim")
    if (
        isinstance(seq_len, bool)
        or isinstance(feature_dim, bool)
        or not isinstance(seq_len, int)
        or not isinstance(feature_dim, int)
    ):
        raise AssertionError("semantic fixture tensorGeometry is invalid")
    return seq_len, feature_dim


__all__ = [
    "DATA_CONTRACT_SHA256",
    "data_contract",
    "fixture_document",
    "model_contract",
    "semantic_digests",
    "tensor_geometry",
]
