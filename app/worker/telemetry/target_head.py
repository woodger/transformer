from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from typing import cast

import torch

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v4 import ModelContract
from app.contracts.target_head_diagnostics.v1.constants import (
    ARTIFACT_FORMAT,
)
from app.worker.model.transformer import TransformerModel, public_predictions


class TargetHeadDiagnosticsCollector:
    """Собирает best-effort наблюдения выходных головок после каждой эпохи."""

    def __init__(self, model_contract: ModelContract) -> None:
        self._model_contract = model_contract
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

    @property
    def failed(self) -> bool:
        return self._failed

    def begin_epoch(self) -> None:
        if self._failed:
            return
        target_width = self._model_contract.target_width
        self._gradient_sums = [0.0] * target_width
        self._gradient_maxima = [0.0] * target_width
        self._gradient_counts = [0] * target_width

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
            head_sum: torch.Tensor | None = None
            with torch.no_grad(), autocast():
                for batch_features in features():
                    if batch_features.size(0) == 0:
                        continue
                    model_output, shared = transformer.forward_with_shared_representation(
                        batch_features.to(device)
                    )
                    raw_logits = model_output[:, :target_width].detach().double()
                    predictions = public_predictions(
                        model_output,
                        self._model_contract,
                    ).detach().double()
                    shared_values = shared.detach().double()
                    _observe_distributions(summaries, raw_logits, predictions)
                    batch_head_sum = shared_values.sum(dim=0).detach().cpu()
                    head_sum = (
                        batch_head_sum
                        if head_sum is None
                        else head_sum + batch_head_sum
                    )

            if _row_count(summaries) != expected_rows or head_sum is None:
                raise ValueError(
                    "target head diagnostics row count differs from committed input"
                )
            head_mean = (head_sum / expected_rows).to(device)
            centered_l2_sum = 0.0
            centered_rows = 0
            with torch.no_grad(), autocast():
                for batch_features in features():
                    if batch_features.size(0) == 0:
                        continue
                    _model_output, shared = transformer.forward_with_shared_representation(
                        batch_features.to(device)
                    )
                    centered = shared.detach().double() - head_mean
                    distances = torch.sqrt(centered.square().sum(dim=1))
                    centered_l2_sum += float(distances.sum().detach().cpu())
                    centered_rows += distances.size(0)

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
            self._epochs.append(cast(JsonObject, {
                "epoch": epoch,
                "globalStep": global_step,
                "headInput": {
                    "rowCenteredL2Mean": centered_l2_sum / expected_rows,
                },
                "targetHeads": cast(list[object], target_heads),
            }))
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
                "target-head-diagnostics-v1\\0"
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


def _target_head(
    model: torch.nn.Module,
    target_width: int,
) -> torch.nn.Linear:
    transformer = _transformer_model(model)
    target_head = transformer.head.target_head
    if target_head.out_features != target_width:
        raise ValueError("target head width differs from model contract")
    return target_head


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
