from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.contracts.model_topology.v2.constants import MAX_EDGES, MAX_NODES
from app.contracts.semantic.v4 import ModelContract
from app.contracts.worker.v17.config import ModelConfig
from app.service.domain.json_types import JsonObject, JsonValue
from app.service.domain.records import PublishedModelRecord


class ModelTopologyBuilder:
    """Материализует публичную topology исполнения одной опубликованной модели."""

    def build(self, model: PublishedModelRecord) -> JsonObject:
        contract = ModelContract.from_document(model.model_contract)
        model_config = ModelConfig.from_manifest(
            _mapping(model.metadata, "modelConfig")
        )
        model_definition_sha256 = _string(
            model.semantic_digests,
            "modelDefinitionSha256",
        )
        target_slots = contract.target_slots
        direct_components = contract.direct_components
        auxiliary_components = contract.auxiliary_components
        resources = contract.resource_declarations
        expected_nodes = (
            6
            + model_config.layers
            + 4 * len(target_slots)
            + len(contract.weighted_binary_target_indices)
            + len(resources)
            + len(direct_components)
            + len(auxiliary_components)
        )
        if expected_nodes > MAX_NODES:
            raise OverflowError("model topology exceeds the node limit")

        nodes: list[JsonObject] = []
        edges: list[JsonObject] = []
        target_index = {
            str(slot["identity"]): index
            for index, slot in enumerate(target_slots)
        }
        target_nodes: dict[str, dict[str, str]] = {}
        resource_nodes: dict[str, str] = {}
        feature_shape = _shape(("batch", None), ("sequence", model_config.seq_len), (
            "feature",
            model_config.feature_dim,
        ))
        context_feature_dim = (
            model_config.feature_dim * 2
            if model_config.context_mode == "relaxed"
            else model_config.feature_dim
        )
        context_shape = _shape(
            ("batch", None),
            ("sequence", model_config.seq_len),
            ("contextFeature", context_feature_dim),
        )
        attention_mask_shape = _shape(
            ("batch", None),
            ("sequence", model_config.seq_len),
        )
        encoded_shape = _shape(
            ("batch", None),
            ("sequence", model_config.seq_len),
            ("hidden", model_config.hidden),
        )
        shared_shape = _shape(("batch", None), ("hidden", model_config.hidden))
        observation_shape = _shape(("batch", None))
        scalar_shape: list[JsonObject] = []

        _append_node(
            nodes,
            _node(
                "features-input",
                "Feature sequence",
                output_ports=("values", "Features"),
            ),
        )
        _append_node(
            nodes,
            _node(
                "context-preparation",
                "Context preparation",
                input_ports=("features", "Features"),
                output_ports=(
                    ("prepared", "Prepared features"),
                    ("attention-mask", "Attention mask"),
                ),
            ),
        )
        _append_edge(
            edges,
            "features-input",
            "values",
            "context-preparation",
            "features",
            feature_shape,
            "notApplicable",
        )
        _append_node(
            nodes,
            _node(
                "input-projection",
                "Input projection",
                input_ports=("features", "Prepared features"),
                output_ports=("encoded", "Encoded sequence"),
            ),
        )
        _append_edge(
            edges,
            "context-preparation",
            "prepared",
            "input-projection",
            "features",
            context_shape,
            "notApplicable",
        )
        _append_node(
            nodes,
            _node(
                "positional-encoding",
                "Positional encoding",
                input_ports=("encoded", "Encoded sequence"),
                output_ports=("positioned", "Positioned sequence"),
            ),
        )
        _append_edge(
            edges,
            "input-projection",
            "encoded",
            "positional-encoding",
            "encoded",
            encoded_shape,
            "propagates",
        )

        previous_node = "positional-encoding"
        previous_port = "positioned"
        for layer in range(1, model_config.layers + 1):
            node_id = f"encoder-{layer}"
            _append_node(
                nodes,
                _node(
                    node_id,
                    f"Encoder layer {layer}",
                    input_ports=(
                        ("sequence", "Encoded sequence"),
                        ("attention-mask", "Attention mask"),
                    ),
                    output_ports=("sequence", "Encoded sequence"),
                    encoderLayer=layer,
                    attentionHeadCount=model_config.nhead,
                ),
            )
            _append_edge(
                edges,
                previous_node,
                previous_port,
                node_id,
                "sequence",
                encoded_shape,
                "propagates",
            )
            _append_edge(
                edges,
                "context-preparation",
                "attention-mask",
                node_id,
                "attention-mask",
                attention_mask_shape,
                "notApplicable",
            )
            previous_node = node_id
            previous_port = "sequence"

        _append_node(
            nodes,
            _node(
                "last-valid-selection",
                "Last valid state",
                input_ports=("sequence", "Encoded sequence"),
                output_ports=("state", "Selected state"),
            ),
        )
        _append_edge(
            edges,
            previous_node,
            previous_port,
            "last-valid-selection",
            "sequence",
            encoded_shape,
            "propagates",
        )

        for index, slot in enumerate(target_slots):
            identity = str(slot["identity"])
            observed_id = f"observed-target-{index}"
            raw_id = f"target-{index}-raw"
            loss_transformation_id = f"target-{index}-loss-input"
            prediction_transformation_id = f"target-{index}-prediction"
            correction_id = f"target-{index}-public-correction"
            target_nodes[identity] = {
                "observed": observed_id,
                "loss": loss_transformation_id,
                "prediction": prediction_transformation_id,
            }
            prediction_source_id = raw_id
            prediction_source_port = "raw"
            positive_class_weight = contract.positive_class_weight_for_target(index)
            _append_node(
                nodes,
                _node(
                    observed_id,
                    f"Observed target {index}",
                    output_ports=("value", "Observed value"),
                    targetIdentity=identity,
                    targetIndex=index,
                ),
            )
            _append_node(
                nodes,
                _node(
                    raw_id,
                    f"Target head {index}",
                    input_ports=("state", "Selected state"),
                    output_ports=("raw", "Raw coordinate"),
                    targetIdentity=identity,
                    targetIndex=index,
                ),
            )
            _append_edge(
                edges,
                "last-valid-selection",
                "state",
                raw_id,
                "state",
                shared_shape,
                "propagates",
            )
            if positive_class_weight is not None:
                _append_node(
                    nodes,
                    _node(
                        correction_id,
                        f"Positive-class logit correction {index}",
                        input_ports=("raw", "Raw coordinate"),
                        output_ports=("corrected", "Corrected logit"),
                        targetIdentity=identity,
                        targetIndex=index,
                        positiveClassWeight=positive_class_weight,
                        derivedCorrection="subtractLogPositiveClassWeight",
                    ),
                )
                _append_edge(
                    edges,
                    raw_id,
                    "raw",
                    correction_id,
                    "raw",
                    observation_shape,
                    "propagates",
                )
                prediction_source_id = correction_id
                prediction_source_port = "corrected"
            _append_node(
                nodes,
                _node(
                    loss_transformation_id,
                    f"Loss input transformation {index}",
                    input_ports=("raw", "Raw coordinate"),
                    output_ports=("value", "Loss estimate"),
                    targetIdentity=identity,
                    targetIndex=index,
                    transformation=str(slot["lossInputTransformation"]),
                ),
            )
            _append_edge(
                edges,
                raw_id,
                "raw",
                loss_transformation_id,
                "raw",
                observation_shape,
                "propagates",
            )
            _append_node(
                nodes,
                _node(
                    prediction_transformation_id,
                    f"Public prediction transformation {index}",
                    input_ports=("raw", "Raw coordinate"),
                    output_ports=("value", "Public prediction"),
                    targetIdentity=identity,
                    targetIndex=index,
                    transformation=str(slot["publicPredictionTransformation"]),
                ),
            )
            _append_edge(
                edges,
                prediction_source_id,
                prediction_source_port,
                prediction_transformation_id,
                "raw",
                observation_shape,
                "propagates",
            )

        for index, resource in enumerate(resources):
            identity = str(resource["identity"])
            node_id = f"resource-{index}"
            resource_nodes[identity] = node_id
            _append_node(
                nodes,
                _node(
                    node_id,
                    f"Private resource {index}",
                    input_ports=("state", "Selected state"),
                    output_ports=("value", "Private value"),
                    resourceIdentity=identity,
                    resourceClass=str(resource["resourceClass"]),
                ),
            )
            _append_edge(
                edges,
                "last-valid-selection",
                "state",
                node_id,
                "state",
                shared_shape,
                "propagates",
            )

        aggregate_ports: list[tuple[str, str]] = []
        for index, component in enumerate(direct_components):
            component_id = str(component["identity"])
            target_identity = str(component["targetIdentity"])
            target_slot_index = target_index[target_identity]
            node_id = f"direct-component-{index}"
            aggregate_port = f"component-{index}"
            aggregate_ports.append((aggregate_port, f"Component {index}"))
            _append_node(
                nodes,
                _node(
                    node_id,
                    f"Direct loss {index}",
                    input_ports=(
                        ("estimate", "Loss estimate"),
                        ("observed", "Observed value"),
                    ),
                    output_ports=(
                        ("mean", "Unweighted mean"),
                        ("weighted", "Weighted contribution"),
                    ),
                    componentIdentity=component_id,
                    operator=str(component["operator"]),
                    componentWeight=float(cast(int | float, component["weight"])),
                    reduction="GlobalRowMean",
                    targetIdentity=target_identity,
                    targetIndex=target_slot_index,
                ),
            )
            target = target_nodes[target_identity]
            _append_edge(
                edges,
                target["loss"],
                "value",
                node_id,
                "estimate",
                observation_shape,
                "propagates",
            )
            _append_edge(
                edges,
                target["observed"],
                "value",
                node_id,
                "observed",
                observation_shape,
                "notApplicable",
            )
            _append_edge(
                edges,
                node_id,
                "weighted",
                "total-loss",
                aggregate_port,
                scalar_shape,
                "propagates",
            )

        direct_count = len(direct_components)
        for index, component in enumerate(auxiliary_components):
            component_id = str(component["identity"])
            operator = str(component["operator"])
            node_id = f"auxiliary-component-{index}"
            aggregate_port = f"component-{direct_count + index}"
            aggregate_ports.append((aggregate_port, f"Component {direct_count + index}"))
            role_ports = _auxiliary_role_ports(operator)
            _append_node(
                nodes,
                _node(
                    node_id,
                    f"Auxiliary loss {index}",
                    input_ports=role_ports,
                    output_ports=(
                        ("mean", "Unweighted mean"),
                        ("weighted", "Weighted contribution"),
                    ),
                    componentIdentity=component_id,
                    operator=operator,
                    componentWeight=float(cast(int | float, component["weight"])),
                    reduction="GlobalRowMean",
                ),
            )
            roles = _mapping(component, "roles")
            for role, _label in role_ports:
                source_node, source_port, gradient_flow = _auxiliary_role_source(
                    operator,
                    role,
                    roles,
                    target_nodes,
                    resource_nodes,
                )
                _append_edge(
                    edges,
                    source_node,
                    source_port,
                    node_id,
                    role,
                    observation_shape,
                    gradient_flow,
                )
            _append_edge(
                edges,
                node_id,
                "weighted",
                "total-loss",
                aggregate_port,
                scalar_shape,
                "propagates",
            )

        _append_node(
            nodes,
            _node(
                "total-loss",
                "Total loss",
                input_ports=tuple(aggregate_ports),
                output_ports=("total", "Total loss"),
                aggregation="WeightedSum",
            ),
        )

        _validate_graph(nodes, edges)

        return cast(JsonObject, {
            "modelDefinitionSha256": model_definition_sha256,
            "topologyRevision": 2,
            "nodes": [cast(JsonValue, node) for node in nodes],
            "edges": [cast(JsonValue, edge) for edge in edges],
        })


def _node(
    node_id: str,
    label: str,
    *,
    input_ports: tuple[str, str] | tuple[tuple[str, str], ...] = (),
    output_ports: tuple[str, str] | tuple[tuple[str, str], ...] = (),
    **attributes: JsonValue,
) -> JsonObject:
    return cast(JsonObject, {
        "id": node_id,
        "label": label,
        "inputPorts": [
            cast(JsonValue, port) for port in _ports(input_ports)
        ],
        "outputPorts": [
            cast(JsonValue, port) for port in _ports(output_ports)
        ],
        **attributes,
    })


def _ports(
    values: tuple[str, str] | tuple[tuple[str, str], ...],
) -> list[JsonObject]:
    if len(values) == 2 and all(isinstance(value, str) for value in values):
        pairs = (cast(tuple[str, str], values),)
    else:
        pairs = cast(tuple[tuple[str, str], ...], values)
    return [{"id": port_id, "label": label} for port_id, label in pairs]


def _shape(
    *axes: tuple[str, int | None],
) -> list[JsonObject]:
    return [{"axis": axis, "size": size} for axis, size in axes]


def _append_node(nodes: list[JsonObject], node: JsonObject) -> None:
    nodes.append(node)
    if len(nodes) > MAX_NODES:
        raise OverflowError("model topology exceeds the node limit")


def _append_edge(
    edges: list[JsonObject],
    source_node_id: str,
    source_port_id: str,
    destination_node_id: str,
    destination_port_id: str,
    shape: list[JsonObject],
    gradient_flow: str,
) -> None:
    edges.append(cast(JsonObject, {
        "from": {"nodeId": source_node_id, "portId": source_port_id},
        "to": {"nodeId": destination_node_id, "portId": destination_port_id},
        "logicalShape": [cast(JsonValue, axis) for axis in shape],
        "gradientFlow": gradient_flow,
    }))
    if len(edges) > MAX_EDGES:
        raise OverflowError("model topology exceeds the edge limit")


def _validate_graph(nodes: list[JsonObject], edges: list[JsonObject]) -> None:
    """Отклоняет внутренне противоречивый публичный graph до публикации."""

    input_ports: dict[str, set[str]] = {}
    output_ports: dict[str, set[str]] = {}
    for node in nodes:
        node_id = cast(str, node["id"])
        if node_id in input_ports:
            raise ValueError("model topology has a duplicate node")
        input_ports[node_id] = _port_ids(node, "inputPorts")
        output_ports[node_id] = _port_ids(node, "outputPorts")

    for edge in edges:
        source = cast(JsonObject, edge["from"])
        destination = cast(JsonObject, edge["to"])
        source_node_id = cast(str, source["nodeId"])
        source_port_id = cast(str, source["portId"])
        destination_node_id = cast(str, destination["nodeId"])
        destination_port_id = cast(str, destination["portId"])
        if source_port_id not in output_ports.get(source_node_id, set()):
            raise ValueError("model topology has an unknown source port")
        if destination_port_id not in input_ports.get(destination_node_id, set()):
            raise ValueError("model topology has an unknown destination port")


def _port_ids(node: JsonObject, field: str) -> set[str]:
    values = cast(list[JsonObject], node[field])
    port_ids = {cast(str, value["id"]) for value in values}
    if len(port_ids) != len(values):
        raise ValueError("model topology has a duplicate port")
    return port_ids


def _auxiliary_role_ports(operator: str) -> tuple[tuple[str, str], ...]:
    if operator == "GaussianNLL":
        return (
            ("locationEstimate", "Location estimate"),
            ("observedLocation", "Observed location"),
            ("scale", "Scale"),
        )
    if operator == "ExpectedValue":
        return (
            ("positiveOutcomeProbability", "Positive outcome probability"),
            ("negativeOutcomeProbability", "Negative outcome probability"),
        )
    if operator == "RiskAdjustedExpectedValue":
        return (
            ("positiveOutcomeProbability", "Positive outcome probability"),
            ("negativeOutcomeProbability", "Negative outcome probability"),
            ("uncertaintyScale", "Uncertainty scale"),
        )
    raise ValueError("auxiliary operator is unavailable for topology")


def _auxiliary_role_source(
    operator: str,
    role: str,
    roles: Mapping[str, object],
    targets: Mapping[str, Mapping[str, str]],
    resources: Mapping[str, str],
) -> tuple[str, str, str]:
    identity = _string(roles, role)
    if role == "observedLocation":
        return targets[identity]["observed"], "value", "notApplicable"
    if role == "locationEstimate":
        return targets[identity]["loss"], "value", "propagates"
    if role in {"positiveOutcomeProbability", "negativeOutcomeProbability"}:
        return targets[identity]["prediction"], "value", "propagates"
    if role in {"scale", "uncertaintyScale"}:
        return (
            resources[identity],
            "value",
            "stopped" if operator == "RiskAdjustedExpectedValue" else "propagates",
        )
    raise ValueError("auxiliary role is unavailable for topology")


def _mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    candidate = value.get(field)
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{field} must be an object")
    mapping = cast(Mapping[object, object], candidate)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{field} keys must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


def _string(value: Mapping[str, object], field: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str):
        raise ValueError(f"{field} must be a string")
    return candidate


__all__ = ["ModelTopologyBuilder"]
