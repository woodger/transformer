from dataclasses import dataclass
import html
import json
import math
import os


PLOT_METRICS = [
    "loss",
    "loss_ret",
    "loss_prob",
    "loss_ev",
    "loss_vol",
    "sigma_min",
    "sigma_p05",
    "sigma_mean",
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
    "week",
    "elapsed_ms",
]


@dataclass
class TrainMetrics:
    rows: int = 0
    batches: int = 0
    loss: float = 0.0
    loss_ret: float = 0.0
    loss_prob: float = 0.0
    loss_ev: float = 0.0
    loss_vol: float = 0.0
    sigma_min: float = 0.0
    sigma_p05: float = 0.0
    sigma_mean: float = 0.0
    grad_norm: float = 0.0
    nan_ratio: float = 0.0
    masked_token_ratio: float = 0.0
    complete_token_ratio: float = 0.0
    partial_token_ratio: float = 0.0
    empty_token_ratio: float = 0.0
    elapsed_ms: float = 0.0
    step: int = 0
    lr: float = 0.0
    week: int = 0

    def update(
        self,
        rows: int,
        loss_parts: dict[str, float],
        grad_norm: float,
        nan_ratio: float,
        masked_token_ratio: float = 0.0,
        complete_token_ratio: float = 0.0,
        partial_token_ratio: float = 0.0,
        empty_token_ratio: float = 0.0,
    ):
        total_rows = self.rows + rows
        if total_rows <= 0:
            return

        def avg(current: float, value: float) -> float:
            return ((current * self.rows) + (value * rows)) / total_rows

        self.loss = avg(self.loss, loss_parts["loss"])
        self.loss_ret = avg(self.loss_ret, loss_parts["loss_ret"])
        self.loss_prob = avg(self.loss_prob, loss_parts["loss_prob"])
        self.loss_ev = avg(self.loss_ev, loss_parts["loss_ev"])
        self.loss_vol = avg(self.loss_vol, loss_parts["loss_vol"])
        self.sigma_min = avg(self.sigma_min, loss_parts.get("sigma_min", 0.0))
        self.sigma_p05 = avg(self.sigma_p05, loss_parts.get("sigma_p05", 0.0))
        self.sigma_mean = avg(self.sigma_mean, loss_parts.get("sigma_mean", 0.0))
        self.grad_norm = avg(self.grad_norm, grad_norm)
        self.nan_ratio = avg(self.nan_ratio, nan_ratio)
        self.masked_token_ratio = avg(self.masked_token_ratio, masked_token_ratio)
        self.complete_token_ratio = avg(self.complete_token_ratio, complete_token_ratio)
        self.partial_token_ratio = avg(self.partial_token_ratio, partial_token_ratio)
        self.empty_token_ratio = avg(self.empty_token_ratio, empty_token_ratio)
        self.rows = total_rows
        self.batches += 1
        self.step = int(loss_parts.get("step", self.step))
        self.week = int(loss_parts.get("week", self.week))

    def log_line(self, **extra) -> str:
        fields = {
            **extra,
            "loss": f"{self.loss:.6f}",
            "ret": f"{self.loss_ret:.6f}",
            "prob": f"{self.loss_prob:.6f}",
            "ev": f"{self.loss_ev:.6f}",
            "vol": f"{self.loss_vol:.6f}",
            "sigma_min": f"{self.sigma_min:.6g}",
            "sigma_p05": f"{self.sigma_p05:.6g}",
            "sigma_mean": f"{self.sigma_mean:.6g}",
            "grad": f"{self.grad_norm:.3f}",
            "rows": self.rows,
            "batches": self.batches,
            "nan": f"{self.nan_ratio:.4f}",
            "masked_tokens": f"{self.masked_token_ratio:.4f}",
            "complete_tokens": f"{self.complete_token_ratio:.4f}",
            "partial_tokens": f"{self.partial_token_ratio:.4f}",
            "empty_tokens": f"{self.empty_token_ratio:.4f}",
            "step": self.step,
            "lr": f"{self.lr:.6g}",
            "week": self.week,
            "ms": f"{self.elapsed_ms:.0f}",
        }

        return " ".join(f"{key}={value}" for key, value in fields.items())

    def to_dict(self, **extra) -> dict:
        return {
            **extra,
            "rows": self.rows,
            "batches": self.batches,
            "loss": self.loss,
            "loss_ret": self.loss_ret,
            "loss_prob": self.loss_prob,
            "loss_ev": self.loss_ev,
            "loss_vol": self.loss_vol,
            "sigma_min": self.sigma_min,
            "sigma_p05": self.sigma_p05,
            "sigma_mean": self.sigma_mean,
            "grad_norm": self.grad_norm,
            "nan_ratio": self.nan_ratio,
            "masked_token_ratio": self.masked_token_ratio,
            "complete_token_ratio": self.complete_token_ratio,
            "partial_token_ratio": self.partial_token_ratio,
            "empty_token_ratio": self.empty_token_ratio,
            "elapsed_ms": self.elapsed_ms,
            "step": self.step,
            "lr": self.lr,
            "week": self.week,
        }

    def __float__(self) -> float:
        return self.loss

    def __format__(self, format_spec: str) -> str:
        return format(self.loss, format_spec)

    def __gt__(self, other) -> bool:
        return self.loss > float(other)


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

        x = row.get("epoch", row.get("frame", index))
        if not isinstance(x, (int, float)):
            x = index
        x = float(x)
        if not math.isfinite(x):
            x = float(index)

        points.append((x, value))

    return points


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


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

    with open(path, "w", encoding="utf-8") as f:
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
