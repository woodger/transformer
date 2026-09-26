from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from typing import cast

import torch

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.semantic.v4 import ModelContract
from app.contracts.target_head_diagnostics.v3.constants import ARTIFACT_FORMAT
from app.worker.model.transformer import TransformerModel, public_predictions

_ENCODER_GROUP_NAMES = ("attention", "feedForward", "normalization")


class TargetHeadDiagnosticsCollector:
    """Собирает best-effort наблюдения выходных головок после каждой эпохи."""

    def __init__(
        self,
        model_contract: ModelContract,
        *,
        collect_encoder_learning: bool = False,
    ) -> None:
        self._model_contract = model_contract
        self._collect_encoder_learning = collect_encoder_learning
        self._layout = tuple(
            {
                "targetIdentity": identity,
                "targetIndex": index,
                "directComponentIdentity": str(component["identity"]),
            }
            for index, (identity, component) in enumerate(
                zip(
                    model_contract.target_identities,
                    model_contract.direct_components,
                    strict=True,
                )
            )
        )
        self._component_identities = tuple(
            str(component["identity"])
            for component in model_contract.direct_components
        )
        self._failed = False
        self._epochs: list[JsonObject] = []
        self._gradient_sums: list[float] = []
        self._gradient_maxima: list[float] = []
        self._gradient_counts: list[int] = []
        self._encoder_parameter_groups: tuple[
            tuple[tuple[torch.nn.Parameter, ...], ...],
            ...,
        ] = ()
        self._encoder_gradient_sums: list[list[list[float]]] = []
        self._encoder_gradient_maxima: list[list[list[float]]] = []
        self._encoder_gradient_counts: list[list[list[int]]] = []
        self._encoder_update_sums: list[list[float]] = []
        self._encoder_update_maxima: list[list[float]] = []
        self._encoder_update_counts: list[list[int]] = []
        self._encoder_parameter_snapshots: tuple[
            tuple[tuple[torch.Tensor, ...], ...],
            ...,
        ] | None = None

    @property
    def failed(self) -> bool:
        return self._failed

    def begin_epoch(self, model: torch.nn.Module | None = None) -> None:
        if self._failed:
            return
        target_width = self._model_contract.target_width
        self._gradient_sums = [0.0] * target_width
        self._gradient_maxima = [0.0] * target_width
        self._gradient_counts = [0] * target_width
        self._encoder_parameter_snapshots = None
        if not self._collect_encoder_learning:
            return
        if model is None:
            raise ValueError("encoder diagnostics require a Transformer model")
        self._encoder_parameter_groups = _encoder_parameter_groups(model)
        layer_count = len(self._encoder_parameter_groups)
        component_count = len(self._component_identities)
        self._encoder_gradient_sums = _new_encoder_gradient_statistics(
            layer_count,
            component_count,
        )
        self._encoder_gradient_maxima = _new_encoder_gradient_statistics(
            layer_count,
            component_count,
        )
        self._encoder_gradient_counts = _new_encoder_gradient_counts(
            layer_count,
            component_count,
        )
        self._encoder_update_sums = _new_encoder_update_statistics(layer_count)
        self._encoder_update_maxima = _new_encoder_update_statistics(layer_count)
        self._encoder_update_counts = _new_encoder_update_counts(layer_count)

    def observe_component_gradients(
        self,
        components: tuple[tuple[str, torch.Tensor], ...],
        model: torch.nn.Module,
    ) -> None:
        if self._failed:
            return
        target_head = _target_head(model, self._model_contract.target_width)
        component_losses = dict(components)
        if (
            len(component_losses) != len(components)
            or not set(self._component_identities).issubset(component_losses)
        ):
            raise ValueError("direct component gradients differ from target layout")

        norms: list[torch.Tensor] = []
        for target_index, component_identity in enumerate(
            self._component_identities
        ):
            weight_gradient, bias_gradient = torch.autograd.grad(
                component_losses[component_identity],
                (target_head.weight, target_head.bias),
                retain_graph=True,
                create_graph=False,
            )
            row_weight_gradient = weight_gradient[target_index].detach().float()
            row_bias_gradient = bias_gradient[target_index].detach().float()
            norms.append(torch.sqrt(
                row_weight_gradient.square().sum()
                + row_bias_gradient.square()
            ))

        values = cast(
            list[float],
            torch.stack(norms).detach().cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
        )
        for index, value in enumerate(values):
            if not math.isfinite(value):
                continue
            self._gradient_sums[index] += value
            self._gradient_maxima[index] = max(
                self._gradient_maxima[index],
                value,
            )
            self._gradient_counts[index] += 1
        if self._collect_encoder_learning:
            self._observe_encoder_component_gradients(component_losses, model)

    def snapshot_encoder_parameters(self, model: torch.nn.Module) -> None:
        if self._failed or not self._collect_encoder_learning:
            return
        parameter_groups = self._matching_encoder_parameter_groups(model)
        self._encoder_parameter_snapshots = tuple(
            tuple(
                tuple(parameter.detach().clone() for parameter in group)
                for group in layer_groups
            )
            for layer_groups in parameter_groups
        )

    def observe_encoder_parameter_updates(
        self,
        model: torch.nn.Module,
        *,
        optimizer_update_applied: bool,
    ) -> None:
        if self._failed or not self._collect_encoder_learning:
            return
        snapshots = self._encoder_parameter_snapshots
        self._encoder_parameter_snapshots = None
        if snapshots is None:
            raise ValueError("encoder parameter snapshots are unavailable")
        if not optimizer_update_applied:
            return
        parameter_groups = self._matching_encoder_parameter_groups(model)
        values = _parameter_update_norm_values(parameter_groups, snapshots)
        value_index = 0
        for layer_index in range(len(parameter_groups)):
            for group_index in range(len(_ENCODER_GROUP_NAMES)):
                value = values[value_index]
                value_index += 1
                if not math.isfinite(value):
                    continue
                self._encoder_update_sums[layer_index][group_index] += value
                self._encoder_update_maxima[layer_index][group_index] = max(
                    self._encoder_update_maxima[layer_index][group_index],
                    value,
                )
                self._encoder_update_counts[layer_index][group_index] += 1

    def _observe_encoder_component_gradients(
        self,
        component_losses: Mapping[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> None:
        parameter_groups = self._matching_encoder_parameter_groups(model)
        parameters = tuple(
            parameter
            for layer_groups in parameter_groups
            for group in layer_groups
            for parameter in group
        )
        for component_index, component_identity in enumerate(
            self._component_identities
        ):
            gradients = torch.autograd.grad(
                component_losses[component_identity],
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
                materialize_grads=False,
            )
            values = _gradient_norm_values(parameter_groups, gradients)
            value_index = 0
            for layer_index in range(len(parameter_groups)):
                for group_index in range(len(_ENCODER_GROUP_NAMES)):
                    value = values[value_index]
                    value_index += 1
                    if not math.isfinite(value):
                        continue
                    self._encoder_gradient_sums[layer_index][component_index][
                        group_index
                    ] += value
                    self._encoder_gradient_maxima[layer_index][component_index][
                        group_index
                    ] = max(
                        self._encoder_gradient_maxima[layer_index][
                            component_index
                        ][group_index],
                        value,
                    )
                    self._encoder_gradient_counts[layer_index][component_index][
                        group_index
                    ] += 1

    def _matching_encoder_parameter_groups(
        self,
        model: torch.nn.Module,
    ) -> tuple[tuple[tuple[torch.nn.Parameter, ...], ...], ...]:
        parameter_groups = _encoder_parameter_groups(model)
        if not _same_parameter_groups(
            self._encoder_parameter_groups,
            parameter_groups,
        ):
            raise ValueError("encoder parameter groups differ within an epoch")
        return parameter_groups

    def _encoder_learning_document(self) -> JsonObject:
        if not self._encoder_parameter_groups:
            raise ValueError("encoder learning statistics are unavailable")
        layers: list[JsonObject] = []
        for layer_index in range(len(self._encoder_parameter_groups)):
            direct_component_gradients: list[JsonObject] = []
            for component_index, component_identity in enumerate(
                self._component_identities
            ):
                direct_component_gradients.append({
                    "directComponentIdentity": component_identity,
                    "groups": _gradient_groups_document(
                        self._encoder_gradient_sums[layer_index][component_index],
                        self._encoder_gradient_maxima[layer_index][component_index],
                        self._encoder_gradient_counts[layer_index][component_index],
                    ),
                })
            layers.append({
                "layerIndex": layer_index,
                "directComponentGradients": cast(
                    list[JsonValue],
                    direct_component_gradients,
                ),
                "parameterUpdates": _parameter_update_groups_document(
                    self._encoder_update_sums[layer_index],
                    self._encoder_update_maxima[layer_index],
                    self._encoder_update_counts[layer_index],
                ),
            })
        return {"layers": cast(list[JsonValue], layers)}

    def observe_post_update(
        self,
        model: torch.nn.Module,
        features: Callable[[], Iterable[torch.Tensor]],
        *,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager[object]],
        epoch: int,
        global_step: int,
        expected_rows: int,
    ) -> None:
        if self._failed:
            return
        if epoch != len(self._epochs) + 1:
            raise ValueError("target head diagnostic epochs are not consecutive")
        if expected_rows < 1:
            raise ValueError("target head diagnostics require committed rows")

        transformer = _transformer_model(model)
        target_width = self._model_contract.target_width
        was_training = transformer.training
        transformer.eval()
        try:
            summaries = _new_distribution_statistics(target_width)
            representation_sums: list[torch.Tensor] | None = None
            with torch.no_grad(), autocast():
                for batch_features in features():
                    if batch_features.size(0) == 0:
                        continue
                    (
                        model_output,
                        shared,
                        encoder_input,
                        encoder_layers,
                    ) = transformer.forward_with_representation_flow(
                        batch_features.to(device)
                    )
                    raw_logits = model_output[:, :target_width].detach().double()
                    predictions = public_predictions(
                        model_output,
                        self._model_contract,
                    ).detach().double()
                    _observe_distributions(summaries, raw_logits, predictions)
                    representations = (
                        encoder_input,
                        *encoder_layers,
                        shared,
                    )
                    batch_sums = [
                        values.detach().double().sum(dim=0).detach().cpu()
                        for values in representations
                    ]
                    if representation_sums is None:
                        representation_sums = batch_sums
                    elif len(representation_sums) != len(batch_sums):
                        raise ValueError(
                            "target head diagnostic representation count differs"
                        )
                    else:
                        representation_sums = [
                            total + batch_sum
                            for total, batch_sum in zip(
                                representation_sums,
                                batch_sums,
                                strict=True,
                            )
                        ]

            if (
                _row_count(summaries) != expected_rows
                or representation_sums is None
            ):
                raise ValueError(
                    "target head diagnostics row count differs from committed input"
                )
            representation_means = tuple(
                (total / expected_rows).to(device)
                for total in representation_sums
            )
            representation_centered_l2_sums = [
                0.0
                for _mean in representation_means
            ]
            centered_rows = 0
            with torch.no_grad(), autocast():
                for batch_features in features():
                    if batch_features.size(0) == 0:
                        continue
                    (
                        _model_output,
                        shared,
                        encoder_input,
                        encoder_layers,
                    ) = transformer.forward_with_representation_flow(
                        batch_features.to(device)
                    )
                    representations = (
                        encoder_input,
                        *encoder_layers,
                        shared,
                    )
                    if len(representations) != len(representation_means):
                        raise ValueError(
                            "target head diagnostic representation count differs"
                        )
                    for index, (values, mean) in enumerate(
                        zip(
                            representations,
                            representation_means,
                            strict=True,
                        )
                    ):
                        centered = values.detach().double() - mean
                        distances = torch.sqrt(centered.square().sum(dim=1))
                        representation_centered_l2_sums[index] += float(
                            distances.sum().detach().cpu()
                        )
                    centered_rows += shared.size(0)

            if centered_rows != expected_rows:
                raise ValueError(
                    "target head diagnostics row count differs between passes"
                )

            target_head = _target_head(transformer, target_width)
            parameters = cast(
                list[float],
                torch.cat((
                    target_head.weight.detach().float().square().sum(dim=1).sqrt(),
                    target_head.bias.detach().float(),
                )).cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            if not all(math.isfinite(value) for value in parameters):
                raise ValueError("target head parameters are non-finite")

            raw_distributions = _distribution_items(summaries, "raw")
            public_distributions = _distribution_items(summaries, "public")
            target_heads: list[JsonObject] = []
            for index, raw_logit in enumerate(raw_distributions):
                count = self._gradient_counts[index]
                target_heads.append({
                    "rawLogit": raw_logit,
                    "publicPrediction": public_distributions[index],
                    "gradientL2Mean": (
                        None
                        if count == 0
                        else self._gradient_sums[index] / count
                    ),
                    "gradientL2Maximum": (
                        None
                        if count == 0
                        else self._gradient_maxima[index]
                    ),
                    "gradientBatchCount": count,
                    "weightL2AfterEpoch": float(parameters[index]),
                    "biasAfterEpoch": float(parameters[target_width + index]),
                })
            observation: JsonObject = {
                "epoch": epoch,
                "globalStep": global_step,
                "representationFlow": _representation_flow(
                    representation_centered_l2_sums,
                    expected_rows,
                ),
                "targetHeads": cast(list[JsonValue], target_heads),
            }
            if self._collect_encoder_learning:
                observation["encoderLearning"] = self._encoder_learning_document()
            self._epochs.append(observation)
        finally:
            transformer.train(was_training)

    def disable(self) -> bool:
        if self._failed:
            return False
        self._failed = True
        return True

    def recovery_document(self) -> JsonObject:
        return {
            "failed": self._failed,
            "epochs": [dict(epoch) for epoch in self._epochs],
        }

    def load_recovery_document(self, document: object) -> None:
        if not isinstance(document, Mapping):
            raise ValueError("target head diagnostics recovery state is invalid")
        values = cast(Mapping[object, object], document)
        if set(values) != {"failed", "epochs"}:
            raise ValueError("target head diagnostics recovery fields are invalid")
        failed = values["failed"]
        epochs = values["epochs"]
        if not isinstance(failed, bool) or not isinstance(epochs, list):
            raise ValueError("target head diagnostics recovery state is invalid")
        raw_epochs = cast(list[object], epochs)
        if not all(isinstance(epoch, dict) for epoch in raw_epochs):
            raise ValueError("target head diagnostics recovery epochs are invalid")
        epoch_documents = [
            cast(dict[object, object], epoch)
            for epoch in raw_epochs
        ]
        self._failed = failed
        self._epochs = [
            cast(JsonObject, dict(epoch))
            for epoch in epoch_documents
        ]

    def artifact(
        self,
        *,
        job_id: str,
        attempt: int,
        attempt_id: str,
        input_revision: int,
        manifest_sha256: str,
        model_definition_sha256: str,
        job_config_sha256: str,
        completed_epochs: int,
    ) -> JsonObject | None:
        if self._failed or len(self._epochs) != completed_epochs:
            return None
        if completed_epochs < 1:
            return None
        sample_identity = hashlib.sha256(
            (
                "target-head-diagnostics-v3\\0"
                + manifest_sha256
                + "\\0"
                + str(input_revision)
            ).encode("ascii")
        ).hexdigest()
        sample_rows = _sample_rows(self._epochs)
        return {
            "format": ARTIFACT_FORMAT,
            "jobId": job_id,
            "attempt": attempt,
            "attemptId": attempt_id,
            "inputRevision": input_revision,
            "manifestSha256": manifest_sha256,
            "modelDefinitionSha256": model_definition_sha256,
            "jobConfigSha256": job_config_sha256,
            "sampleIdentity": f"thds_{sample_identity}",
            "sampleRowCount": sample_rows,
            "layout": [dict(item) for item in self._layout],
            "coverage": {"completedEpochs": completed_epochs},
            "epochs": [dict(epoch) for epoch in self._epochs],
        }


def _encoder_parameter_groups(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[torch.nn.Parameter, ...], ...], ...]:
    transformer = _transformer_model(model)
    layers: list[tuple[tuple[torch.nn.Parameter, ...], ...]] = []
    for layer in transformer.encoder.layers:
        if not isinstance(layer, torch.nn.TransformerEncoderLayer):
            raise ValueError("encoder diagnostics require TransformerEncoderLayer")
        attention = _trainable_parameters(layer.self_attn)
        feed_forward = (
            *_trainable_parameters(layer.linear1),
            *_trainable_parameters(layer.linear2),
        )
        normalization = (
            *_trainable_parameters(layer.norm1),
            *_trainable_parameters(layer.norm2),
        )
        groups = (attention, feed_forward, normalization)
        if any(not group for group in groups):
            raise ValueError("encoder diagnostics parameter group is empty")
        all_parameters = _trainable_parameters(layer)
        grouped_parameters = tuple(
            parameter
            for group in groups
            for parameter in group
        )
        if (
            len({id(parameter) for parameter in grouped_parameters})
            != len(grouped_parameters)
            or {id(parameter) for parameter in grouped_parameters}
            != {id(parameter) for parameter in all_parameters}
        ):
            raise ValueError("encoder diagnostics parameter groups are incomplete")
        layers.append(groups)
    if not layers:
        raise ValueError("encoder diagnostics require encoder layers")
    return tuple(layers)


def _trainable_parameters(
    module: torch.nn.Module,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _same_parameter_groups(
    expected: tuple[tuple[tuple[torch.nn.Parameter, ...], ...], ...],
    actual: tuple[tuple[tuple[torch.nn.Parameter, ...], ...], ...],
) -> bool:
    return len(expected) == len(actual) and all(
        len(expected_layer) == len(actual_layer)
        and all(
            len(expected_group) == len(actual_group)
            and all(
                expected_parameter is actual_parameter
                for expected_parameter, actual_parameter in zip(
                    expected_group,
                    actual_group,
                    strict=True,
                )
            )
            for expected_group, actual_group in zip(
                expected_layer,
                actual_layer,
                strict=True,
            )
        )
        for expected_layer, actual_layer in zip(expected, actual, strict=True)
    )


def _gradient_norm_values(
    parameter_groups: tuple[tuple[tuple[torch.nn.Parameter, ...], ...], ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> list[float]:
    gradient_index = 0
    norms: list[torch.Tensor] = []
    for layer_groups in parameter_groups:
        for group in layer_groups:
            group_gradients = gradients[
                gradient_index : gradient_index + len(group)
            ]
            gradient_index += len(group)
            norms.append(_gradient_l2(group_gradients, group[0]))
    if gradient_index != len(gradients):
        raise ValueError("encoder gradient count differs from parameter groups")
    return _host_values(norms)


def _gradient_l2(
    gradients: tuple[torch.Tensor | None, ...],
    reference: torch.nn.Parameter,
) -> torch.Tensor:
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        return reference.detach().new_zeros((), dtype=torch.float32)
    return torch.stack(squares).sum().sqrt()


def _parameter_update_norm_values(
    parameter_groups: tuple[tuple[tuple[torch.nn.Parameter, ...], ...], ...],
    snapshots: tuple[tuple[tuple[torch.Tensor, ...], ...], ...],
) -> list[float]:
    if len(parameter_groups) != len(snapshots):
        raise ValueError("encoder parameter snapshots differ from parameter groups")
    norms: list[torch.Tensor] = []
    for layer_groups, layer_snapshots in zip(
        parameter_groups,
        snapshots,
        strict=True,
    ):
        if len(layer_groups) != len(layer_snapshots):
            raise ValueError("encoder parameter snapshots differ from parameter groups")
        for group, group_snapshots in zip(
            layer_groups,
            layer_snapshots,
            strict=True,
        ):
            if len(group) != len(group_snapshots):
                raise ValueError("encoder parameter snapshots differ from parameter groups")
            squares = [
                (parameter.detach().float() - snapshot.float()).square().sum()
                for parameter, snapshot in zip(group, group_snapshots, strict=True)
            ]
            norms.append(torch.stack(squares).sum().sqrt())
    return _host_values(norms)


def _host_values(values: list[torch.Tensor]) -> list[float]:
    return cast(
        list[float],
        torch.stack(values).detach().cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
    )


def _new_encoder_gradient_statistics(
    layer_count: int,
    component_count: int,
) -> list[list[list[float]]]:
    return [
        [[0.0] * len(_ENCODER_GROUP_NAMES) for _ in range(component_count)]
        for _ in range(layer_count)
    ]


def _new_encoder_gradient_counts(
    layer_count: int,
    component_count: int,
) -> list[list[list[int]]]:
    return [
        [[0] * len(_ENCODER_GROUP_NAMES) for _ in range(component_count)]
        for _ in range(layer_count)
    ]


def _new_encoder_update_statistics(layer_count: int) -> list[list[float]]:
    return [[0.0] * len(_ENCODER_GROUP_NAMES) for _ in range(layer_count)]


def _new_encoder_update_counts(layer_count: int) -> list[list[int]]:
    return [[0] * len(_ENCODER_GROUP_NAMES) for _ in range(layer_count)]


def _gradient_groups_document(
    sums: list[float],
    maxima: list[float],
    counts: list[int],
) -> JsonObject:
    return {
        name: {
            "l2Mean": None if count == 0 else sums[index] / count,
            "l2Maximum": None if count == 0 else maxima[index],
            "finiteBatchCount": count,
        }
        for index, (name, count) in enumerate(
            zip(_ENCODER_GROUP_NAMES, counts, strict=True)
        )
    }


def _parameter_update_groups_document(
    sums: list[float],
    maxima: list[float],
    counts: list[int],
) -> JsonObject:
    return {
        name: {
            "l2Mean": None if count == 0 else sums[index] / count,
            "l2Maximum": None if count == 0 else maxima[index],
            "appliedBatchCount": count,
        }
        for index, (name, count) in enumerate(
            zip(_ENCODER_GROUP_NAMES, counts, strict=True)
        )
    }


def _target_head(
    model: torch.nn.Module,
    target_width: int,
) -> torch.nn.Linear:
    transformer = _transformer_model(model)
    target_head = transformer.head.target_head
    if target_head.out_features != target_width:
        raise ValueError("target head width differs from model contract")
    return target_head


def _representation_flow(
    centered_l2_sums: list[float],
    row_count: int,
) -> JsonObject:
    if len(centered_l2_sums) < 3:
        raise ValueError("target head diagnostics require encoder layers")
    return {
        "encoderInput": {
            "rowCenteredL2Mean": centered_l2_sums[0] / row_count,
        },
        "encoderLayers": [
            {
                "layerIndex": index,
                "rowCenteredL2Mean": value / row_count,
            }
            for index, value in enumerate(centered_l2_sums[1:-1])
        ],
        "targetHeadInput": {
            "rowCenteredL2Mean": centered_l2_sums[-1] / row_count,
        },
    }


def _transformer_model(model: torch.nn.Module) -> TransformerModel:
    if not isinstance(model, TransformerModel):
        raise ValueError("target head diagnostics require a Transformer model")
    return model


def _new_distribution_statistics(target_width: int) -> dict[str, object]:
    return {
        "rowCount": 0,
        "rawMinimum": [float("inf")] * target_width,
        "rawMaximum": [float("-inf")] * target_width,
        "rawSum": [0.0] * target_width,
        "rawSquareSum": [0.0] * target_width,
        "publicMinimum": [float("inf")] * target_width,
        "publicMaximum": [float("-inf")] * target_width,
        "publicSum": [0.0] * target_width,
        "publicSquareSum": [0.0] * target_width,
    }


def _observe_distributions(
    statistics: dict[str, object],
    raw_logits: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    target_width = raw_logits.size(1)
    if predictions.shape != raw_logits.shape:
        raise ValueError("public prediction shape differs from raw logits")
    if raw_logits.size(0) == 0:
        return
    values = cast(
        list[float],
        torch.cat((
            raw_logits.min(dim=0).values,
            raw_logits.max(dim=0).values,
            raw_logits.sum(dim=0),
            raw_logits.square().sum(dim=0),
            predictions.min(dim=0).values,
            predictions.max(dim=0).values,
            predictions.sum(dim=0),
            predictions.square().sum(dim=0),
        )).detach().cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target head diagnostics contain non-finite values")
    _update_distribution_values(statistics, "raw", values[: 4 * target_width])
    _update_distribution_values(statistics, "public", values[4 * target_width :])
    statistics["rowCount"] = _row_count(statistics) + raw_logits.size(0)


def _update_distribution_values(
    statistics: dict[str, object],
    prefix: str,
    values: list[float],
) -> None:
    target_width = len(values) // 4
    minimum = _float_list(statistics, f"{prefix}Minimum")
    maximum = _float_list(statistics, f"{prefix}Maximum")
    total = _float_list(statistics, f"{prefix}Sum")
    square_total = _float_list(statistics, f"{prefix}SquareSum")
    for index in range(target_width):
        minimum[index] = min(minimum[index], values[index])
        maximum[index] = max(maximum[index], values[target_width + index])
        total[index] += values[2 * target_width + index]
        square_total[index] += values[3 * target_width + index]


def _distribution_items(
    statistics: dict[str, object],
    prefix: str,
) -> tuple[JsonObject, ...]:
    row_count = _row_count(statistics)
    if row_count < 1:
        raise ValueError("target head diagnostics have no rows")
    minimum = _float_list(statistics, f"{prefix}Minimum")
    maximum = _float_list(statistics, f"{prefix}Maximum")
    total = _float_list(statistics, f"{prefix}Sum")
    square_total = _float_list(statistics, f"{prefix}SquareSum")
    values: list[JsonObject] = []
    for index in range(len(minimum)):
        mean = total[index] / row_count
        variance = max(0.0, square_total[index] / row_count - mean * mean)
        values.append({
            "rowCount": row_count,
            "minimum": minimum[index],
            "maximum": maximum[index],
            "mean": mean,
            "standardDeviation": math.sqrt(variance),
        })
    return tuple(values)


def _sample_rows(epochs: list[JsonObject]) -> int:
    first_epoch = epochs[0]
    target_heads = cast(list[JsonObject], first_epoch["targetHeads"])
    row_count = cast(JsonObject, target_heads[0]["rawLogit"])["rowCount"]
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise ValueError("target head diagnostics sample rows are invalid")
    return row_count


def _row_count(statistics: Mapping[str, object]) -> int:
    value = statistics["rowCount"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("target head diagnostics row count is invalid")
    return value


def _float_list(
    statistics: Mapping[str, object],
    name: str,
) -> list[float]:
    value = statistics[name]
    if not isinstance(value, list):
        raise ValueError("target head diagnostic distribution is invalid")
    items = cast(list[object], value)
    if not all(isinstance(item, float) for item in items):
        raise ValueError("target head diagnostic distribution is invalid")
    return cast(list[float], value)


__all__ = ["TargetHeadDiagnosticsCollector"]
