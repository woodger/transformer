from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from app.contracts.flight.v4.arrow import (
    TARGET_WIDTH,
    validate_target_space_values,
)
from app.contracts.json_types import JsonObject
from app.contracts.worker.v6.objective import DIRECT_LOSSES

TARGET_NAMES = tuple(semantic for _, semantic, _ in DIRECT_LOSSES)


def _empty_int_set() -> set[int]:
    return set()


@dataclass(slots=True)
class TargetStatisticsAccumulator:
    """Aggregate one immutable fit dataset exactly once per input ordinal."""

    _count: int = 0
    _ordinals: set[int] = field(default_factory=_empty_int_set)
    _minimum: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.full(TARGET_WIDTH, math.inf, dtype=np.float64)
    )
    _maximum: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.full(TARGET_WIDTH, -math.inf, dtype=np.float64)
    )
    _mean: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(TARGET_WIDTH, dtype=np.float64)
    )
    _m2: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(TARGET_WIDTH, dtype=np.float64)
    )
    _zero_count: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(TARGET_WIDTH, dtype=np.int64)
    )
    _one_count: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(TARGET_WIDTH, dtype=np.int64)
    )

    @property
    def count(self) -> int:
        return self._count

    def contains(self, ordinal: int) -> bool:
        return ordinal in self._ordinals

    def update(self, ordinal: int, targets: torch.Tensor) -> None:
        if ordinal < 0:
            raise ValueError("target statistics require a non-negative ordinal")
        if ordinal in self._ordinals:
            return
        if targets.ndim != 2 or targets.shape[1] != TARGET_WIDTH:
            raise ValueError(
                f"target statistics require shape [rows, {TARGET_WIDTH}]"
            )
        if targets.dtype != torch.float32:
            raise ValueError("target statistics require float32 values")
        if targets.device.type != "cpu":
            raise ValueError("target statistics require a CPU tensor")
        if targets.shape[0] == 0:
            return

        values = np.asarray(targets.detach().numpy(), dtype=np.float32)
        validate_target_space_values(values)
        values64 = values.astype(np.float64, copy=False)
        batch_count = int(values64.shape[0])
        batch_mean = values64.mean(axis=0)
        centered = values64 - batch_mean
        batch_m2 = np.asarray(
            np.sum(centered * centered, axis=0, dtype=np.float64),
            dtype=np.float64,
        )

        if self._count == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
        else:
            total = self._count + batch_count
            delta = batch_mean - self._mean
            self._mean = self._mean + delta * (batch_count / total)
            updated_m2 = np.asarray(
                self._m2
                + batch_m2
                + delta * delta * (self._count * batch_count / total),
                dtype=np.float64,
            )
            self._m2 = updated_m2

        self._minimum = np.minimum(self._minimum, values64.min(axis=0))
        self._maximum = np.maximum(self._maximum, values64.max(axis=0))
        self._zero_count += np.count_nonzero(values == 0.0, axis=0)
        self._one_count += np.count_nonzero(values == 1.0, axis=0)
        self._count += batch_count
        self._ordinals.add(ordinal)

    def to_documents(self) -> list[JsonObject]:
        if self._count <= 0:
            raise ValueError("target statistics require a non-empty dataset")
        variances = np.maximum(self._m2 / self._count, 0.0)
        standard_deviations = np.sqrt(variances)
        documents: list[JsonObject] = []
        for index, name in enumerate(TARGET_NAMES):
            values = (
                float(self._minimum[index]),
                float(self._maximum[index]),
                float(self._mean[index]),
                float(standard_deviations[index]),
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("target statistics must be finite")
            documents.append({
                "targetIndex": index,
                "name": name,
                "count": self._count,
                "min": values[0],
                "max": values[1],
                "mean": values[2],
                "std": values[3],
                "zeroCount": int(self._zero_count[index]),
                "oneCount": int(self._one_count[index]),
            })
        return documents


__all__ = ["TARGET_NAMES", "TargetStatisticsAccumulator"]
