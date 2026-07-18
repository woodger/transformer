import html
import math
import os

from app.metrics.io import load_metrics_jsonl
from app.storage.atomic import atomic_output_path


PLOT_METRICS = [
    "loss",
    "loss_ret",
    "loss_prob",
    "loss_ev",
    "loss_vol",
    "sigma_min",
    "sigma_p05",
    "sigma_mean",
    "ret_mae",
    "ret_rmse",
    "ret_mae_baseline",
    "ret_mae_skill",
    "ret_mae_improvement",
    "monitor_value",
    "best_monitor",
    "grad_norm",
    "nan_ratio",
    "masked_token_ratio",
    "complete_token_ratio",
    "partial_token_ratio",
    "empty_token_ratio",
    "rows",
    "batches",
    "step",
    "lr",
    "loss_stage",
    "elapsed_ms",
]


def plot_metrics(jsonl_path: str, output_dir: str) -> list[str]:
    rows = load_metrics_jsonl(jsonl_path)
    if not rows:
        raise ValueError("metrics JSONL is empty")

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for metric in PLOT_METRICS:
        points = _series(rows, metric)
        if not points:
            continue

        path = os.path.join(output_dir, f"{metric}.svg")
        _write_svg(path, points, metric)
        paths.append(path)

    return paths


def _series(rows: list[dict], metric: str) -> list[tuple[float, float]]:
    points = []
    for index, row in enumerate(rows, start=1):
        value = row.get(metric)
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


def _write_svg(path: str, points: list[tuple[float, float]], title: str):
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
