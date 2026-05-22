import json
import math
import os

from app.metrics.types import TrainMetrics


def reset_metrics_log(path: str):
    if path is None:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8"):
        pass


def append_metrics_jsonl(path: str, metrics: TrainMetrics, **extra):
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


def load_metrics_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
