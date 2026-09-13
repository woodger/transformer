from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NoReturn, cast

from app.contracts.json_types import JsonValue
from app.contracts.semantic.v2.constants import (
    AUXILIARY_OPERATORS,
    DIRECT_OPERATORS,
    MAX_OBJECTIVE_COMPONENTS,
    MAX_PRIVATE_RESOURCES,
    MAX_TARGET_SLOTS,
    MODEL_ARCHITECTURE_IDENTITY,
    MODEL_ARCHITECTURE_REVISION,
    OBJECTIVE_LANGUAGE_REVISION,
)
from app.contracts.semantic.v2.schema import SemanticSchemaError, validate_schema


class SemanticContractError(ValueError):
    def __init__(self, reason: str, path: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = path


def validate_model_contract(document: object) -> None:
    try:
        validate_schema(document, "model-contract-envelope")
    except SemanticSchemaError as exc:
        raise SemanticContractError(
            "INVALID_MODEL_CONTRACT",
            exc.path,
            str(exc),
        ) from exc

    model = _mapping(document)
    revision = model.get("objectiveLanguageRevision")
    if revision != OBJECTIVE_LANGUAGE_REVISION:
        raise SemanticContractError(
            "LANGUAGE_REVISION_UNAVAILABLE",
            "/objectiveLanguageRevision",
            f"objective language revision {revision!r} is unavailable",
        )

    try:
        validate_schema(document, "model-contract")
    except SemanticSchemaError as exc:
        raise SemanticContractError(
            _schema_reason(exc.path),
            exc.path,
            str(exc),
        ) from exc

    _validate_finite_json(cast(JsonValue, document))
    _validate_target_contract(model)
    _validate_objective(model)
    _validate_model_config(model)


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
        constraint = _mapping(slot["observedConstraint"])
        constraint_name = constraint["constraint"]
        if constraint_name == "ClosedInterval":
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
    resources = _mapping_sequence(objective["resources"])
    direct = _mapping_sequence(objective["directComponents"])
    auxiliary = _mapping_sequence(objective["auxiliaryComponents"])

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
    if operator not in DIRECT_OPERATORS:
        _fail(
            "UNKNOWN_PRIMITIVE",
            f"/objective/directComponents/{index}/operator",
            f"unknown direct operator: {operator}",
        )
    roles = _mapping(component["roles"])
    identity = str(slot["identity"])
    if operator == "BinaryCrossEntropyWithLogits":
        estimate_role, observed_role = "logit", "probability"
    else:
        estimate_role, observed_role = "estimate", "observed"
    if not _target_ref(roles[estimate_role], identity, "lossEstimate"):
        _fail(
            "INVALID_OBJECTIVE",
            f"/objective/directComponents/{index}/roles/{estimate_role}",
            "direct estimate reference does not match its target slot",
        )
    if not _target_ref(roles[observed_role], identity, "observed"):
        _fail(
            "INVALID_OBJECTIVE",
            f"/objective/directComponents/{index}/roles/{observed_role}",
            "direct observed reference does not match its target slot",
        )

    constraint = _mapping(slot["observedConstraint"])
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
    if operator not in AUXILIARY_OPERATORS:
        _fail(
            "UNKNOWN_PRIMITIVE",
            f"/objective/auxiliaryComponents/{index}/operator",
            f"unknown auxiliary operator: {operator}",
        )
    roles = _mapping(component["roles"])
    path = f"/objective/auxiliaryComponents/{index}/roles"
    if operator == "GaussianNLL":
        location = _target_identity(roles["locationEstimate"], "lossEstimate")
        observed = _target_identity(roles["observedLocation"], "observed")
        resource = _resource_identity(roles["scale"])
        if location != observed or location not in slots or resource not in resources:
            _fail(
                "INVALID_OBJECTIVE",
                path,
                "GaussianNLL roles do not resolve to one target and one resource",
            )
        return {resource}, {resource}

    positive = _target_identity(
        roles["positiveOutcomeProbability"],
        "publicPrediction",
    )
    negative = _target_identity(
        roles["negativeOutcomeProbability"],
        "publicPrediction",
    )
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
    resource = _resource_identity(roles["uncertaintyScale"])
    if resource not in resources:
        _fail(
            "INVALID_RESOURCE_GRAPH",
            path,
            "RiskAdjustedExpectedValue resource is not declared",
        )
    return {resource}, set()


def _validate_model_config(model: Mapping[str, object]) -> None:
    config = _mapping(model["modelConfig"])
    architecture = _mapping(config["architecture"])
    if (
        architecture["identity"] != MODEL_ARCHITECTURE_IDENTITY
        or architecture["revision"] != MODEL_ARCHITECTURE_REVISION
    ):
        _fail(
            "UNKNOWN_PRIMITIVE",
            "/modelConfig/architecture",
            "model architecture is not implemented",
        )
    if cast(int, config["hidden"]) % cast(int, config["nhead"]):
        _fail(
            "INVALID_MODEL_CONTRACT",
            "/modelConfig",
            "hidden must be divisible by nhead",
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


def _target_ref(value: object, identity: str, view: str) -> bool:
    reference = _mapping(value)
    return (
        reference.get("targetIdentity") == identity
        and reference.get("view") == view
    )


def _target_identity(value: object, view: str) -> str:
    reference = _mapping(value)
    if reference.get("view") != view:
        return ""
    return str(reference.get("targetIdentity", ""))


def _resource_identity(value: object) -> str:
    reference = _mapping(value)
    return str(reference.get("resourceIdentity", ""))


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


def _number(value: object) -> float:
    return float(cast(int | float, value))


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _fail(reason: str, path: str, message: str) -> NoReturn:
    raise SemanticContractError(reason, path, message)


__all__ = ["SemanticContractError", "validate_model_contract"]
