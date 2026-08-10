import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.args import parse_args


def get_device(value):
    from app.runtime.device import get_device as implementation

    return implementation(value)


def configure_reproducibility(seed, deterministic):
    from app.runtime.reproducibility import configure_reproducibility as implementation

    return implementation(seed, deterministic)


def build_model(*args, **kwargs):
    from app.training.factory import build_model as implementation

    return implementation(*args, **kwargs)


def build_trainer(*args, **kwargs):
    from app.training.factory import build_trainer as implementation

    return implementation(*args, **kwargs)


def fit_stream(args, device):
    from app.commands.fit_stream import run

    return run(args, device, build_model, build_trainer)


def predict_stream(args, device):
    from app.commands.predict_stream import run

    return run(args, device, build_model, build_trainer)


def run_fit(args, device, model_factory, trainer_factory):
    from app.commands.fit import run

    return run(args, device, model_factory, trainer_factory)


def run_predict(args, device, model_factory, trainer_factory):
    from app.commands.predict import run

    return run(args, device, model_factory, trainer_factory)


def run_plot_metrics(args):
    from app.commands.plot_metrics import run

    return run(args)


def run_gmark(args):
    from app.commands.gmark import run

    return run(args)


def reset_metrics_log(path):
    from app.metrics import reset_metrics_log as implementation

    return implementation(path)


def resolve_metrics_path(name):
    from app.utils import resolve_metrics_path as implementation

    return implementation(name)


def main():
    args = parse_args()

    if args.action == "flight" and args.flight_action == "serve":
        from app.service.bootstrap.application import run_from_args

        run_from_args(args)
        return

    if args.action == "auth":
        from app.admin.bootstrap.auth_tokens import run

        run(args)
        return

    if args.action == "db":
        from app.admin.bootstrap.db_migrations import run

        run(args)
        return

    if args.action == "plot-metrics":
        run_plot_metrics(args)
        return

    if args.action == "gmark":
        raise SystemExit(run_gmark(args))

    if args.action in ("fit", "fit-stream"):
        configure_reproducibility(args.seed, args.deterministic)

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
