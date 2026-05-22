import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.args import parse_args
from app.commands.fit import run as run_fit
from app.commands.fit_stream import run as run_fit_stream
from app.commands.plot_metrics import run as run_plot_metrics
from app.commands.predict import run as run_predict
from app.commands.predict_stream import run as run_predict_stream
from app.runtime.device import get_device
from app.training.factory import build_model, build_trainer
from app.metrics import reset_metrics_log
from app.utils import resolve_metrics_path


def fit_stream(args, device):
    return run_fit_stream(args, device, build_model, build_trainer)


def predict_stream(args, device):
    return run_predict_stream(args, device, build_model, build_trainer)


def main():
    args = parse_args()

    if args.action == "plot-metrics":
        run_plot_metrics(args)
        return

    device = get_device(args.device)

    if args.action == "predict-stream":
        print(f"Using device: {device}", file=sys.stderr)
        if args.data is not None:
            raise ValueError("predict-stream reads stdin; data path is not supported")
        predict_stream(args, device)
        return

    print(f"Using device: {device}")

    if args.action == "fit-stream":
        if args.data is not None:
            raise ValueError("fit-stream reads stdin; data path is not supported")
        reset_metrics_log(resolve_metrics_path(args.metrics_name))
        fit_stream(args, device)
        return

    if args.data is None:
        raise ValueError("data path is required for fit and predict")

    if args.action == "fit":
        reset_metrics_log(resolve_metrics_path(args.metrics_name))
        run_fit(args, device, build_model, build_trainer)
        return

    run_predict(args, device, build_model, build_trainer)


if __name__ == "__main__":
    main()
