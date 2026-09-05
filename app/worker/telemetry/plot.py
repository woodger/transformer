import html
import math
import os
from typing import cast

from app.contracts.json_types import JsonObject
from app.worker.checkpoints.atomic import atomic_output_path
from app.worker.telemetry.io import load_metrics_jsonl

SCALAR_METRICS = (
    "loss",
    "selectionScore",
    "trainingBatchesCompleted",
    "optimizerUpdatesApplied",
    "optimizerUpdatesSkipped",
    "ampOverflowBatches",
    "finiteGradientBatches",
    "nonFiniteGradientBatches",
    "preClipGradientNormMean",
    "preClipGradientNormMax",
    "preClipGradientNormP95",
    "nanRatio",
    "maskedTokenRatio",
    "completeTokenRatio",
    "partialTokenRatio",
    "emptyTokenRatio",
    "rows",
    "batches",
    "step",
    "lr",
    "inputPipelineMs",
    "missingStatsMs",
    "hostToDeviceMs",
    "trainStepMs",
    "elapsedMs",
)


def plot_metrics(jsonl_path: str, output_dir: str) -> list[str]:
    rows = load_metrics_jsonl(jsonl_path)
    if not rows:
        raise ValueError("metrics JSONL is empty")

    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    for metric in _metric_names(rows):
        points = _series(rows, metric)
        if not points:
            continue

        path = os.path.join(output_dir, f"{metric}.svg")
        _write_svg(path, points, metric)
        paths.append(path)

    return paths


def _series(
    rows: list[JsonObject],
    metric: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index, row in enumerate(rows, start=1):
        value = _metric_value(row, metric)
        if not isinstance(value, (int, float)):
            continue
        value = float(value)
        if not math.isfinite(value):
            continue

        x = row.get("step", row.get("epoch", row.get("frame", index)))
        if not isinstance(x, (int, float)):
            x = index
        x = float(x)
        if not math.isfinite(x):
            x = float(index)

        points.append((x, value))

    return points


def _metric_value(row: JsonObject, metric: str) -> object:
    if metric.startswith("direct."):
        return _named_metric(
            row.get("directLosses"),
            name=metric.removeprefix("direct."),
            name_field="componentIdentity",
            value_field="value",
        )
    if metric.startswith("target."):
        _, target_identity, field = metric.split(".", 2)
        return _named_metric(
            row.get("targetMetrics"),
            name=target_identity,
            name_field="targetIdentity",
            value_field=field,
        )
    if metric.startswith("auxiliary."):
        return _named_metric(
            row.get("auxiliaryLosses"),
            name=metric.removeprefix("auxiliary."),
            name_field="componentIdentity",
            value_field="value",
        )
    if metric.startswith("gradient.component."):
        interactions = row.get("gradientInteractions")
        if not isinstance(interactions, dict):
            return None
        return _named_metric(
            cast(JsonObject, interactions).get("components"),
            name=metric.removeprefix("gradient.component."),
            name_field="componentIdentity",
            value_field="meanNorm",
        )
    if metric.startswith("gradient.pair."):
        interactions = row.get("gradientInteractions")
        if not isinstance(interactions, dict):
            return None
        left, separator, right = metric.removeprefix("gradient.pair.").partition(
            "__"
        )
        if not separator:
            return None
        pairs = cast(JsonObject, interactions).get("pairs")
        if not isinstance(pairs, list):
            return None
        for item in cast(list[object], pairs):
            if not isinstance(item, dict):
                continue
            document = cast(JsonObject, item)
            if (
                document.get("leftComponentIdentity") == left
                and document.get("rightComponentIdentity") == right
            ):
                return document.get("meanCosine")
        return None
    return row.get(metric)


def _metric_names(rows: list[JsonObject]) -> tuple[str, ...]:
    discovered: set[str] = set()
    for row in rows:
        for component in _objects(row.get("directLosses")):
            identity = component.get("componentIdentity")
            if isinstance(identity, str):
                discovered.add(f"direct.{identity}")
        for component in _objects(row.get("auxiliaryLosses")):
            identity = component.get("componentIdentity")
            if isinstance(identity, str):
                discovered.add(f"auxiliary.{identity}")
        for target in _objects(row.get("targetMetrics")):
            identity = target.get("targetIdentity")
            if isinstance(identity, str):
                discovered.add(f"target.{identity}.mae")
                discovered.add(f"target.{identity}.rmse")

        interactions = row.get("gradientInteractions")
        if not isinstance(interactions, dict):
            continue
        document = cast(JsonObject, interactions)
        for component in _objects(document.get("components")):
            name = component.get("componentIdentity")
            if isinstance(name, str):
                discovered.add(f"gradient.component.{name}")
        for pair in _objects(document.get("pairs")):
            left = pair.get("leftComponentIdentity")
            right = pair.get("rightComponentIdentity")
            if isinstance(left, str) and isinstance(right, str):
                discovered.add(f"gradient.pair.{left}__{right}")
    return (*SCALAR_METRICS, *sorted(discovered))


def _named_metric(
    value: object,
    *,
    name: str,
    name_field: str,
    value_field: str,
) -> object:
    for document in _objects(value):
        if document.get(name_field) == name:
            return document.get(value_field)
    return None


def _objects(value: object) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        cast(JsonObject, item)
        for item in cast(list[object], value)
        if isinstance(item, dict)
    )


def _write_svg(
    path: str,
    points: list[tuple[float, float]],
    title: str,
) -> None:
    width = 960
    height = 420
    left = 68
    right = 24
    top = 36
    bottom = 58

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        padding = abs(min_y) * 0.1 or 1.0
        min_y -= padding
        max_y += padding

    plot_w = width - left - right
    plot_h = height - top - bottom

    def scale_x(value: float) -> float:
        return left + ((value - min_x) / (max_x - min_x)) * plot_w

    def scale_y(value: float) -> float:
        return top + plot_h - ((value - min_y) / (max_y - min_y)) * plot_h

    polyline = " ".join(
        f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points
    )
    title_text = html.escape(title)

    with atomic_output_path(path) as temporary_path:
        with open(temporary_path, "w", encoding="utf-8") as f:
            f.write(
                f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="24" font-family="monospace" font-size="16" fill="#111111">{title_text}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222222" stroke-width="1"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222222" stroke-width="1"/>
  <text x="{left}" y="{height - 20}" font-family="monospace" font-size="12" fill="#555555">step {min_x:g} -> {max_x:g}</text>
  <text x="8" y="{top + 12}" font-family="monospace" font-size="12" fill="#555555">{max_y:.6g}</text>
  <text x="8" y="{top + plot_h}" font-family="monospace" font-size="12" fill="#555555">{min_y:.6g}</text>
  <polyline fill="none" stroke="#0f766e" stroke-width="2" points="{polyline}"/>
</svg>
"""
            )
