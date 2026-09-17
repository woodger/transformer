from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v3.digests import target_objective_digests
from app.contracts.semantic.v3.validation import validate_model_contract


@dataclass(frozen=True, slots=True)
class ModelContract:
    """Validated target/objective intent and model tuning."""

    document: JsonObject

    @classmethod
    def from_document(cls, document: object) -> ModelContract:
        validate_model_contract(document)
        return cls(cast(JsonObject, deepcopy(document)))

    def to_document(self) -> JsonObject:
        return deepcopy(self.document)

    @property
    def target_contract(self) -> JsonObject:
        return cast(JsonObject, deepcopy(self.document["targetContract"]))

    @property
    def objective(self) -> JsonObject:
        return cast(JsonObject, deepcopy(self.document["objective"]))

    @property
    def model_tuning(self) -> JsonObject:
        return cast(JsonObject, deepcopy(self.document["modelTuning"]))

    @property
    def target_slots(self) -> tuple[JsonObject, ...]:
        target = cast(Mapping[str, object], self.document["targetContract"])
        slots = cast(Sequence[object], target["slots"])
        return tuple(cast(JsonObject, deepcopy(slot)) for slot in slots)

    @property
    def target_identities(self) -> tuple[str, ...]:
        return tuple(cast(str, slot["identity"]) for slot in self.target_slots)

    @property
    def target_width(self) -> int:
        return len(self.target_identities)

    @property
    def resource_declarations(self) -> tuple[JsonObject, ...]:
        objective = cast(Mapping[str, object], self.document["objective"])
        resources = cast(Sequence[object], objective.get("resources", ()))
        return tuple(cast(JsonObject, deepcopy(item)) for item in resources)

    @property
    def direct_components(self) -> tuple[JsonObject, ...]:
        objective = cast(Mapping[str, object], self.document["objective"])
        components = cast(Sequence[object], objective["directComponents"])
        return tuple(cast(JsonObject, deepcopy(item)) for item in components)

    @property
    def auxiliary_components(self) -> tuple[JsonObject, ...]:
        objective = cast(Mapping[str, object], self.document["objective"])
        components = cast(Sequence[object], objective.get("auxiliaryComponents", ()))
        return tuple(cast(JsonObject, deepcopy(item)) for item in components)

    @property
    def direct_loss_weights(self) -> tuple[float, ...]:
        return tuple(
            float(cast(int | float, item["weight"]))
            for item in self.direct_components
        )

    def target_objective_digests(self) -> JsonObject:
        return target_objective_digests(self.document)


__all__ = ["ModelContract"]
