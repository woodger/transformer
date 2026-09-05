import argparse

from app.cli.formatting import COMMAND_EXAMPLES, COMMAND_HELP, HelpFormatter
from app.cli.options import (
    ArgumentTarget,
    add_hidden_help_argument,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    seed,
)
from app.cli.parsers import SubparserTarget
from app.config import DEFAULT_DEVICE, DEFAULT_MAX_FRAME_BYTES
from app.contracts.worker.v12.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DETERMINISTIC,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
)


def add_local_parsers(subparsers: SubparserTarget) -> None:
    _add_fit_parser(subparsers, "fit", stream=False)
    _add_predict_parser(subparsers, "predict", stream=False)
    _add_fit_parser(subparsers, "fit-stream", stream=True)
    _add_predict_parser(subparsers, "predict-stream", stream=True)
    _add_plot_metrics_parser(subparsers)


def _add_device_argument(group: ArgumentTarget) -> None:
    group.add_argument(
        "--device",
        choices=["cpu", "gpu", "auto"],
        default=DEFAULT_DEVICE,
        help="Runtime device. auto uses a GPU when available.",
    )


def _add_max_frame_bytes_argument(group: ArgumentTarget) -> None:
    group.add_argument(
        "--max-frame-bytes",
        type=positive_int,
        default=DEFAULT_MAX_FRAME_BYTES,
        help="Maximum accepted framed Arrow payload size in bytes.",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--lr",
        type=positive_float,
        default=DEFAULT_LR,
        help="Learning rate.",
    )
    train.add_argument(
        "--weight-decay",
        type=nonnegative_float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Adam weight decay.",
    )
    train.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size.",
    )
    train.add_argument(
        "--epochs",
        type=positive_int,
        default=DEFAULT_EPOCHS,
        help="Epoch count.",
    )
    train.add_argument(
        "--select-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Select the best checkpoint by weighted direct loss. Disabled by "
            "default; the last epoch is then published."
        ),
    )
    train.add_argument(
        "--selection-min-delta",
        type=nonnegative_float,
        default=None,
        help="Minimum checkpoint-selection score improvement.",
    )
    train.add_argument(
        "--selection-patience",
        type=nonnegative_int,
        default=None,
        help="Non-improving epochs before early stopping.",
    )
    train.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use mixed precision when running on a GPU.",
    )
    train.add_argument(
        "--seed",
        type=seed,
        default=DEFAULT_SEED,
        help=(
            "Random seed applied to Python, NumPy and PyTorch and saved in "
            "config."
        ),
    )
    train.add_argument(
        "--deterministic",
        action="store_true",
        default=DEFAULT_DETERMINISTIC,
        help=(
            "Request deterministic PyTorch algorithms (unsupported ops may "
            "fail)."
        ),
    )


def _add_fit_parser(
    subparsers: SubparserTarget,
    name: str,
    *,
    stream: bool,
) -> None:
    parser = subparsers.add_parser(
        name,
        add_help=False,
        help=COMMAND_HELP[name],
        epilog=COMMAND_EXAMPLES.get(name),
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(parser)
    if stream:
        parser.set_defaults(data=None)
    else:
        parser.add_argument("data", metavar="INPUT", help="Input Arrow file.")

    runtime = parser.add_argument_group("Runtime")
    _add_device_argument(runtime)
    runtime.add_argument(
        "--checkpoint-out",
        "--model-name",
        dest="model_name",
        default="model_weights.pth",
        help=(
            "Checkpoint output path; relative paths are resolved inside "
            "models/."
        ),
    )
    runtime.add_argument(
        "--metrics-out",
        "--metrics-name",
        dest="metrics_name",
        default=None,
        help=(
            "Metrics JSONL output path; omit to disable metrics logging. "
            "Relative paths are resolved inside models/."
        ),
    )
    if stream:
        _add_max_frame_bytes_argument(runtime)
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model-contract",
        required=True,
        metavar="JSON",
        help="Consumer-neutral ModelContract JSON file.",
    )
    _add_training_arguments(parser)


def _add_predict_parser(
    subparsers: SubparserTarget,
    name: str,
    *,
    stream: bool,
) -> None:
    parser = subparsers.add_parser(
        name,
        add_help=False,
        help=COMMAND_HELP[name],
        epilog=COMMAND_EXAMPLES.get(name),
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(parser)
    if stream:
        parser.set_defaults(data=None)
    else:
        parser.add_argument("data", metavar="INPUT", help="Input Arrow file.")

    runtime = parser.add_argument_group("Runtime")
    _add_device_argument(runtime)
    runtime.add_argument(
        "--checkpoint",
        "--model-name",
        dest="model_name",
        default="model_weights.pth",
        help=(
            "Checkpoint input path, including its model contract; relative "
            "paths are resolved inside models/."
        ),
    )
    runtime.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use mixed precision on a GPU; disabled with a diagnostic on CPU.",
    )
    if stream:
        _add_max_frame_bytes_argument(runtime)
        runtime.add_argument(
            "--pred-col",
            default="out",
            help="Prediction column name.",
        )
    else:
        runtime.add_argument(
            "--output",
            "--preds-path",
            dest="preds_path",
            default="/tmp/preds.arrow",
            help="Arrow output path.",
        )
        runtime.add_argument(
            "--pred-col",
            default="out",
            help="Prediction column name.",
        )


def _add_plot_metrics_parser(subparsers: SubparserTarget) -> None:
    parser = subparsers.add_parser(
        "plot-metrics",
        add_help=False,
        help=COMMAND_HELP["plot-metrics"],
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(parser)
    parser.add_argument(
        "data",
        metavar="METRICS_FILE",
        help="Metrics JSONL file.",
    )
    parser.add_argument(
        "--plots-dir",
        default="metrics_plots",
        help="Directory for generated SVG charts.",
    )
    parser.set_defaults(metrics_name=None)


__all__ = ["add_local_parsers"]
