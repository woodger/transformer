from metrics import plot_metrics
from utils import resolve_metrics_path


def run(args):
    metrics_name = args.data or args.metrics_name
    metrics_path = resolve_metrics_path(metrics_name)
    if metrics_path is None:
        raise ValueError("metrics name is required for plot-metrics")

    paths = plot_metrics(metrics_path, args.plots_dir)
    print(f"Saved {len(paths)} plot(s) to {args.plots_dir}")
