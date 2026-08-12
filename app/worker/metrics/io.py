import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import cast

from app.contracts.json_types import JsonObject, JsonValue
from app.worker.metrics.types import TrainMetrics
from app.worker.runtime.checkpoints.atomic import atomic_output_path


def reset_metrics_log(path: str | None) -> None:
    if path is None:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with atomic_output_path(path) as temporary_path:
        with open(temporary_path, "w", encoding="utf-8"):
            pass


def append_metrics_jsonl(
    path: str | None,
    metrics: TrainMetrics,
    **extra: JsonValue,
) -> None:
    if path is None:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = _json_safe(metrics.to_dict(**extra))
    with open(path, "a", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write("\n")


def load_metrics_jsonl(path: str) -> list[JsonObject]:
    rows: list[JsonObject] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                document: object = json.loads(line)
                value = _json_safe(document)
                if not isinstance(value, dict):
                    raise ValueError("metrics JSONL row must be an object")
                rows.append(value)

    return rows


def _json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("metrics object field names must be strings")
        return {
            cast(str, key): _json_safe(item)
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in cast(Sequence[object], value)]
    raise TypeError(f"metrics value is not JSON-compatible: {type(value).__name__}")
