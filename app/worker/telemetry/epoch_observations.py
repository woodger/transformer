from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.contracts.json_types import JsonObject
from app.contracts.ml import TARGET_IDENTITIES, canonical_targets


def _empty_float_list() -> list[float]:
    return []


def _empty_float_map() -> dict[str, float]:
    return {}


def _empty_integer_map() -> dict[str, int]:
    return {}


@dataclass
class EpochTelemetry:
    """Optional runtime observations for one completed training epoch."""

    targets: tuple[str, ...] = TARGET_IDENTITIES
    target_mae: dict[str, float] = field(default_factory=_empty_float_map)
    target_rmse: dict[str, float] = field(default_factory=_empty_float_map)
    training_batches_completed: int = 0
    optimizer_updates_applied: int = 0
    optimizer_updates_skipped: int = 0
    amp_overflow_batches: int = 0
    finite_gradient_batches: int = 0
    non_finite_gradient_batches: int = 0
    pre_clip_gradient_norm_mean: float | None = None
    pre_clip_gradient_norm_max: float | None = None
    pre_clip_gradient_norm_p95: float | None = None
    nan_ratio: float = 0.0
    masked_token_ratio: float = 0.0
    complete_token_ratio: float = 0.0
    partial_token_ratio: float = 0.0
    empty_token_ratio: float = 0.0
    input_pipeline_ms: float = 0.0
    missing_stats_ms: float = 0.0
    host_to_device_ms: float = 0.0
    train_step_ms: float = 0.0
    elapsed_ms: float = 0.0
    gradient_interaction_samples: int = 0
    _rows: int = field(default=0, repr=False)
    _finite_gradient_norms: list[float] = field(
        default_factory=_empty_float_list,
        repr=False,
    )
    _component_norm_sums: dict[str, float] = field(
        default_factory=_empty_float_map,
        repr=False,
    )
    _component_norm_counts: dict[str, int] = field(
        default_factory=_empty_integer_map,
        repr=False,
    )
    _pair_cosine_sums: dict[str, float] = field(
        default_factory=_empty_float_map,
        repr=False,
    )
    _pair_cosine_counts: dict[str, int] = field(
        default_factory=_empty_integer_map,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.targets = canonical_targets(self.targets)
        self.target_mae = {
            target: float(self.target_mae.get(target, 0.0))
            for target in self.targets
        }
        self.target_rmse = {
            target: float(self.target_rmse.get(target, 0.0))
            for target in self.targets
        }

    def observe_batch(
        self,
        *,
        rows: int,
        target_errors: dict[str, tuple[float, float]],
        grad_norm: float | None,
        optimizer_update_applied: bool,
        amp_overflow: bool,
        nan_ratio: float,
        masked_token_ratio: float = 0.0,
        complete_token_ratio: float = 0.0,
        partial_token_ratio: float = 0.0,
        empty_token_ratio: float = 0.0,
    ) -> None:
        total_rows = self._rows + rows
        if total_rows <= 0:
            return
        if tuple(target_errors) != self.targets:
            raise ValueError("target telemetry differs from objective targets")

        def average(current: float, value: float) -> float:
            return math.fsum((current * self._rows, value * rows)) / total_rows

        for target in self.targets:
            mae, mse = target_errors[target]
            self.target_mae[target] = average(self.target_mae[target], mae)
            combined_mse = math.fsum((
                self.target_rmse[target] ** 2 * self._rows,
                mse * rows,
            )) / total_rows
            self.target_rmse[target] = math.sqrt(max(0.0, combined_mse))

        finite_gradient_norm = (
            grad_norm
            if grad_norm is not None and math.isfinite(grad_norm) and grad_norm >= 0
            else None
        )
        self.training_batches_completed += 1
        if optimizer_update_applied:
            self.optimizer_updates_applied += 1
        else:
            self.optimizer_updates_skipped += 1
        if amp_overflow:
            self.amp_overflow_batches += 1
        if finite_gradient_norm is not None:
            self.finite_gradient_batches += 1
            self._finite_gradient_norms.append(finite_gradient_norm)
        else:
            self.non_finite_gradient_batches += 1
        self.nan_ratio = average(self.nan_ratio, nan_ratio)
        self.masked_token_ratio = average(
            self.masked_token_ratio,
            masked_token_ratio,
        )
        self.complete_token_ratio = average(
            self.complete_token_ratio,
            complete_token_ratio,
        )
        self.partial_token_ratio = average(
            self.partial_token_ratio,
            partial_token_ratio,
        )
        self.empty_token_ratio = average(
            self.empty_token_ratio,
            empty_token_ratio,
        )
        self._rows = total_rows

    def observe_gradient_interactions(self, observation: JsonObject) -> None:
        self.gradient_interaction_samples += 1
        components = observation.get("components")
        pairs = observation.get("pairs")
        if not isinstance(components, list) or not isinstance(pairs, list):
            raise ValueError("gradient interaction observation is invalid")
        for component in components:
            if not isinstance(component, dict):
                raise ValueError("gradient component observation is invalid")
            name = component.get("name")
            norm = component.get("norm")
            if not isinstance(name, str) or not isinstance(norm, (int, float)):
                raise ValueError("gradient component observation is invalid")
            value = float(norm)
            if not math.isfinite(value):
                continue
            self._component_norm_sums[name] = math.fsum((
                self._component_norm_sums.get(name, 0.0),
                value,
            ))
            self._component_norm_counts[name] = (
                self._component_norm_counts.get(name, 0) + 1
            )
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("gradient pair observation is invalid")
            left = pair.get("left")
            right = pair.get("right")
            cosine = pair.get("cosine")
            if (
                not isinstance(left, str)
                or not isinstance(right, str)
                or not isinstance(cosine, (int, float))
            ):
                raise ValueError("gradient pair observation is invalid")
            value = float(cosine)
            if not math.isfinite(value):
                continue
            key = left + "\u0000" + right
            self._pair_cosine_sums[key] = math.fsum((
                self._pair_cosine_sums.get(key, 0.0),
                value,
            ))
            self._pair_cosine_counts[key] = self._pair_cosine_counts.get(key, 0) + 1

    def gradient_interactions_document(self) -> JsonObject | None:
        if self.gradient_interaction_samples == 0:
            return None
        return {
            "samples": self.gradient_interaction_samples,
            "components": [
                {
                    "name": name,
                    "meanNorm": total / self._component_norm_counts[name],
                }
                for name, total in sorted(self._component_norm_sums.items())
            ],
            "pairs": [
                {
                    "left": key.split("\u0000", 1)[0],
                    "right": key.split("\u0000", 1)[1],
                    "meanCosine": total / self._pair_cosine_counts[key],
                }
                for key, total in sorted(self._pair_cosine_sums.items())
            ],
        }

    def finalize_gradient_statistics(self) -> None:
        if self._finite_gradient_norms:
            ordered = sorted(self._finite_gradient_norms)
            self.pre_clip_gradient_norm_mean = math.fsum(ordered) / len(ordered)
            self.pre_clip_gradient_norm_max = ordered[-1]
            nearest_rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
            self.pre_clip_gradient_norm_p95 = ordered[nearest_rank]
        elif self.finite_gradient_batches == 0:
            self.pre_clip_gradient_norm_mean = None
            self.pre_clip_gradient_norm_max = None
            self.pre_clip_gradient_norm_p95 = None


__all__ = ["EpochTelemetry"]
