from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v1 import ModelContract

SEMANTIC_FIXTURES = (
    Path(__file__).parents[2]
    / "app"
    / "contracts"
    / "semantic"
    / "v1"
    / "fixtures"
)
DATA_CONTRACT_SHA256 = "d" * 64


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
    raw_config = contract["modelConfig"]
    if not isinstance(raw_config, dict):
        raise AssertionError("semantic fixture has no modelConfig")
    config = cast(JsonObject, raw_config)
    overrides: tuple[tuple[str, object | None], ...] = (
        ("seqLen", seq_len),
        ("featureDim", feature_dim),
        ("hidden", hidden),
        ("layers", layers),
        ("dropout", dropout),
        ("nhead", nhead),
        ("mode", mode),
    )
    for field, value in overrides:
        if value is not None:
            config[field] = value
    return ModelContract.from_document(contract)


def data_contract(contract: ModelContract) -> JsonObject:
    config = contract.model_config
    return {
        "identity": "test.dataset",
        "revision": 1,
        "profile": "test.profile",
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "seqLen": config["seqLen"],
        "featureDim": config["featureDim"],
    }


def fixture_document(name: str) -> JsonObject:
    path = SEMANTIC_FIXTURES / f"{name}.json"
    with path.open(encoding="utf-8") as source:
        value: object = json.load(source)
    if not isinstance(value, dict):
        raise AssertionError("semantic fixture must be an object")
    return cast(JsonObject, value)


__all__ = [
    "DATA_CONTRACT_SHA256",
    "data_contract",
    "fixture_document",
    "model_contract",
]
