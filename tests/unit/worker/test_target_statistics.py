import math

import pytest
import torch

from app.worker.telemetry import TargetStatisticsAccumulator


def test_target_statistics_use_float32_values_and_population_std():
    targets = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.25, 0.5, 1.0],
            [0.0, 0.5, 1.0, 0.50, 0.5, 0.0],
            [1.0, 1.0, 0.5, 0.75, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    statistics = TargetStatisticsAccumulator()

    statistics.update(0, targets[:1])
    statistics.update(1, targets[1:])
    statistics.update(0, targets[:1])
    documents = statistics.to_documents()

    mean_return = documents[0]
    assert mean_return == {
        "targetIndex": 0,
        "name": "meanReturn",
        "count": 3,
        "min": -1.0,
        "max": 1.0,
        "mean": 0.0,
        "std": pytest.approx(math.sqrt(2.0 / 3.0)),
        "zeroCount": 1,
        "oneCount": 1,
    }
    sigma_return = documents[1]
    assert sigma_return["mean"] == pytest.approx(0.5)
    assert sigma_return["std"] == pytest.approx(math.sqrt(1.0 / 6.0))
    assert documents[4]["std"] == pytest.approx(0.0)


def test_target_statistics_reject_non_float32_or_out_of_range_values():
    statistics = TargetStatisticsAccumulator()
    with pytest.raises(ValueError, match="float32"):
        statistics.update(0, torch.zeros((1, 6), dtype=torch.float64))

    invalid = torch.zeros((1, 6), dtype=torch.float32)
    invalid[0, 2] = 2.0
    with pytest.raises(ValueError, match="outside"):
        statistics.update(0, invalid)
