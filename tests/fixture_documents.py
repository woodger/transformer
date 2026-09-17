from __future__ import annotations

import json
from typing import cast

from app.contracts.json_types import JsonObject
from app.project import PROJECT_ROOT


def semantic_fixture_document(name: str) -> JsonObject:
    path = (
        PROJECT_ROOT
        / "app"
        / "contracts"
        / "semantic"
        / "v3"
        / "fixtures"
        / f"{name}.json"
    )
    with path.open(encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, dict):
        raise AssertionError("semantic fixture must be an object")
    return cast(JsonObject, document)
