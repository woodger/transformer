from typing import Protocol

from app.worker.telemetry import plot_metrics
from app.worker.telemetry.paths import resolve_metrics_path


class PlotMetricsArguments(Protocol):
    data: str | None
    metrics_name: str | None
    plots_dir: str


def run(args: PlotMetricsArguments) -> None:
    metrics_name = args.data or args.metrics_name
    metrics_path = resolve_metrics_path(metrics_name)
    if metrics_path is None:
        raise ValueError("metrics name is required for plot-metrics")

    paths = plot_metrics(metrics_path, args.plots_dir)
    print(f"Saved {len(paths)} plot(s) to {args.plots_dir}")
