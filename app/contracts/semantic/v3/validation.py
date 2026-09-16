from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NoReturn, cast

from app.contracts.json_types import JsonValue
from app.contracts.semantic.v3.constants import (
    MAX_OBJECTIVE_COMPONENTS,
    MAX_PRIVATE_RESOURCES,
    MAX_TARGET_SLOTS,
)
from app.contracts.semantic.v3.schema import SemanticSchemaError, validate_schema


class SemanticContractError(ValueError):
    def __init__(self, reason: str, path: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = path


def validate_model_contract(document: object) -> None:
    try:
        validate_schema(document, "model-contract-envelope")
        validate_schema(document, "model-contract")
    except SemanticSchemaError as exc:
        raise SemanticContractError(
            _schema_reason(exc.path),
            exc.path,
            str(exc),
        ) from exc

    _validate_finite_json(cast(JsonValue, document))
    model = _mapping(document)
    _validate_target_contract(model)
    _validate_objective(model)
    _validate_model_tuning(model)


def _validate_target_contract(model: Mapping[str, object]) -> None:
    target = _mapping(model["targetContract"])
    slots = _mapping_sequence(target["slots"])
    if len(slots) > MAX_TARGET_SLOTS:
        _fail(
            "INVALID_TARGET_CONTRACT",
            "/targetContract/slots",
            f"target slot count exceeds {MAX_TARGET_SLOTS}",
        )
    identities = [str(slot["identity"]) for slot in slots]
    if len(identities) != len(set(identities)):
        _fail(
            "INVALID_TARGET_CONTRACT",
            "/targetContract/slots",
            "target identities must be unique",
        )

    for index, slot in enumerate(slots):
        constraint = _constraint(slot)
        if constraint["constraint"] != "ClosedInterval":
            continue
        minimum = _number(constraint["minimum"])
        maximum = _number(constraint["maximum"])
        if minimum > maximum:
            _fail(
                "INVALID_TARGET_CONTRACT",
                f"/targetContract/slots/{index}/observedConstraint",
                "closed interval minimum exceeds maximum",
            )
        public = str(slot["publicPredictionTransformation"])
        compatible = (
            (public == "Tanh" and minimum <= -1 and maximum >= 1)
            or (public == "Sigmoid" and minimum <= 0 and maximum >= 1)
        )
        if not compatible:
            _fail(
                "INVALID_TARGET_CONTRACT",
                f"/targetContract/slots/{index}/publicPredictionTransformation",
                "public transformation does not prove the observed interval",
            )


def _validate_objective(model: Mapping[str, object]) -> None:
    slots = _mapping_sequence(_mapping(model["targetContract"])["slots"])
    slot_by_identity = {str(slot["identity"]): slot for slot in slots}
    objective = _mapping(model["objective"])
    resources = _mapping_sequence(objective.get("resources", ()))
    direct = _mapping_sequence(objective["directComponents"])
    auxiliary = _mapping_sequence(objective.get("auxiliaryComponents", ()))

    if len(resources) > MAX_PRIVATE_RESOURCES:
        _fail(
            "INVALID_RESOURCE_GRAPH",
            "/objective/resources",
            f"private resource count exceeds {MAX_PRIVATE_RESOURCES}",
        )
    if len(direct) + len(auxiliary) > MAX_OBJECTIVE_COMPONENTS:
        _fail(
            "INVALID_OBJECTIVE",
            "/objective",
            f"objective component count exceeds {MAX_OBJECTIVE_COMPONENTS}",
        )

    resource_identities = [str(item["identity"]) for item in resources]
    if resource_identities != sorted(resource_identities):
        _fail(
            "INVALID_RESOURCE_GRAPH",
            "/objective/resources",
            "resources must use canonical ASCII identity order",
        )
    if len(resource_identities) != len(set(resource_identities)):
        _fail(
            "INVALID_RESOURCE_GRAPH",
            "/objective/resources",
            "resource identities must be unique",
        )
    auxiliary_identities = [str(item["identity"]) for item in auxiliary]
    if auxiliary_identities != sorted(auxiliary_identities):
        _fail(
            "INVALID_OBJECTIVE",
            "/objective/auxiliaryComponents",
            "auxiliary components must use canonical ASCII identity order",
        )
    component_identities = [
        str(item["identity"]) for item in (*direct, *auxiliary)
    ]
    if len(component_identities) != len(set(component_identities)):
        _fail(
            "INVALID_OBJECTIVE",
            "/objective",
            "component identities must be unique",
        )

    if len(direct) != len(slots):
        _fail(
            "INVALID_OBJECTIVE",
            "/objective/directComponents",
            "every target slot must have exactly one direct component",
        )
    for index, (component, slot) in enumerate(zip(direct, slots, strict=True)):
        _validate_direct_component(component, slot, index)

    used_resources: set[str] = set()
    gradient_resources: set[str] = set()
    for index, component in enumerate(auxiliary):
        used, gradient = _validate_auxiliary_component(
            component,
            slot_by_identity,
            set(resource_identities),
            index,
        )
        used_resources.update(used)
        gradient_resources.update(gradient)
    declared = set(resource_identities)
    if used_resources != declared:
        _fail(
            "INVALID_RESOURCE_GRAPH",
            "/objective/resources",
            "every declared resource must be referenced and every reference declared",
        )
    if not declared.issubset(gradient_resources):
        _fail(
            "INVALID_RESOURCE_GRAPH",
            "/objective/resources",
            "every trainable resource requires a gradient-producing path",
        )


def _validate_direct_component(
    component: Mapping[str, object],
    slot: Mapping[str, object],
    index: int,
) -> None:
    operator = str(component["operator"])
    identity = str(slot["identity"])
    if component["targetIdentity"] != identity:
        _fail(
            "INVALID_OBJECTIVE",
            f"/objective/directComponents/{index}/targetIdentity",
            "direct target must resolve to its ordered target slot",
        )

    constraint = _constraint(slot)
    loss_transformation = str(slot["lossInputTransformation"])
    if operator == "BinaryCrossEntropyWithLogits":
        if loss_transformation != "Identity" or not _interval_subset(
            constraint,
            0.0,
            1.0,
        ):
            _fail(
                "INVALID_OBJECTIVE",
                f"/objective/directComponents/{index}",
                "BinaryCrossEntropyWithLogits requires raw logits and observed [0,1]",
            )
    if operator == "LogMSE":
        if loss_transformation != "Sigmoid" or not _nonnegative(constraint):
            _fail(
                "INVALID_OBJECTIVE",
                f"/objective/directComponents/{index}",
                "LogMSE requires a positive estimate and non-negative observed values",
            )


def _validate_auxiliary_component(
    component: Mapping[str, object],
    slots: Mapping[str, Mapping[str, object]],
    resources: set[str],
    index: int,
) -> tuple[set[str], set[str]]:
    operator = str(component["operator"])
    roles = _mapping(component["roles"])
    path = f"/objective/auxiliaryComponents/{index}/roles"
    if operator == "GaussianNLL":
        location = _identity(roles["locationEstimate"])
        observed = _identity(roles["observedLocation"])
        resource = _identity(roles["scale"])
        if location != observed or location not in slots or resource not in resources:
            _fail(
                "INVALID_OBJECTIVE",
                path,
                "GaussianNLL roles do not resolve to one target and one resource",
            )
        return {resource}, {resource}

    positive = _identity(roles["positiveOutcomeProbability"])
    negative = _identity(roles["negativeOutcomeProbability"])
    if (
        positive == negative
        or positive not in slots
        or negative not in slots
        or any(
            str(slots[identity]["publicPredictionTransformation"]) != "Sigmoid"
            for identity in (positive, negative)
        )
    ):
        _fail(
            "INVALID_OBJECTIVE",
            path,
            f"{operator} requires two different public probability slots",
        )
    if operator == "ExpectedValue":
        return set(), set()
    resource = _identity(roles["uncertaintyScale"])
    if resource not in resources:
        _fail(
            "INVALID_RESOURCE_GRAPH",
            path,
            "RiskAdjustedExpectedValue resource is not declared",
        )
    return {resource}, set()


def _validate_model_tuning(model: Mapping[str, object]) -> None:
    tuning = _mapping(model["modelTuning"])
    if _integer(tuning["hiddenWidth"]) % _integer(tuning["attentionHeadCount"]):
        _fail(
            "INVALID_MODEL_CONTRACT",
            "/modelTuning",
            "hiddenWidth must be divisible by attentionHeadCount",
        )


def _validate_finite_json(value: JsonValue, path: str = "") -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            _fail(
                "INVALID_MODEL_CONTRACT",
                path,
                "JSON numbers must be finite IEEE 754 binary64",
            )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}/{_pointer_part(str(key))}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}/{index}")


def _interval_subset(
    constraint: Mapping[str, object],
    minimum: float,
    maximum: float,
) -> bool:
    return (
        constraint.get("constraint") == "ClosedInterval"
        and _number(constraint["minimum"]) >= minimum
        and _number(constraint["maximum"]) <= maximum
    )


def _nonnegative(constraint: Mapping[str, object]) -> bool:
    return (
        constraint.get("constraint") == "ClosedInterval"
        and _number(constraint["minimum"]) >= 0
    )


def _schema_reason(path: str) -> str:
    if path.startswith("/targetContract"):
        return "INVALID_TARGET_CONTRACT"
    if path.startswith("/objective/resources"):
        return "INVALID_RESOURCE_GRAPH"
    if path.startswith("/objective"):
        return "INVALID_OBJECTIVE"
    return "INVALID_MODEL_CONTRACT"


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in cast(Sequence[object], value))


def _identity(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value


def _constraint(slot: Mapping[str, object]) -> Mapping[str, object]:
    value = slot.get("observedConstraint")
    if value is None:
        return {"constraint": "Finite"}
    return _mapping(value)


def _integer(value: object) -> int:
    return int(cast(int | float, value))


def _number(value: object) -> float:
    return float(cast(int | float, value))


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _fail(reason: str, path: str, message: str) -> NoReturn:
    raise SemanticContractError(reason, path, message)


__all__ = ["SemanticContractError", "validate_model_contract"]
