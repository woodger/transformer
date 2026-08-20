import argparse

from app.cli.formatting import COMMAND_EXAMPLES, COMMAND_HELP, HelpFormatter
from app.cli.options import (
    ArgumentTarget,
    add_hidden_help_argument,
    dropout,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    seed,
    six_positive_floats,
)
from app.cli.parsers import SubparserTarget
from app.contracts.worker.v7.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_DETERMINISTIC,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_LOSS_SCHEDULE,
    DEFAULT_LOSS_STAGE,
    DEFAULT_LR,
    DEFAULT_NHEAD,
    DEFAULT_SEED,
    DEFAULT_STAGE_SIZE,
    DEFAULT_WEIGHT_DECAY,
)
from app.local.config import DEFAULT_DEVICE, DEFAULT_MAX_FRAME_BYTES


def add_local_parsers(subparsers: SubparserTarget) -> None:
    _add_fit_parser(subparsers, "fit", stream=False)
    _add_predict_parser(subparsers, "predict", stream=False)
    _add_fit_parser(subparsers, "fit-stream", stream=True)
    _add_predict_parser(subparsers, "predict-stream", stream=True)
    _add_plot_metrics_parser(subparsers)


def _add_device_argument(group: ArgumentTarget) -> None:
    group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_DEVICE,
        help="Runtime device. auto uses CUDA when available.",
    )


def _add_max_frame_bytes_argument(group: ArgumentTarget) -> None:
    group.add_argument(
        "--max-frame-bytes",
        type=positive_int,
        default=DEFAULT_MAX_FRAME_BYTES,
        help="Maximum accepted framed Arrow payload size in bytes.",
    )


def _add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    required_seq_len: bool,
    training: bool,
) -> None:
    model = parser.add_argument_group("Model")
    if required_seq_len:
        model.add_argument(
            "--seq-len",
            type=positive_int,
            required=True,
            default=argparse.SUPPRESS,
            help="Sequence length.",
        )
    else:
        model.add_argument(
            "--seq-len",
            type=positive_int,
            default=None,
            help=(
                "Sequence length; read from the checkpoint. If specified, "
                "it must match."
            ),
        )

    if training:
        defaults = {
            "hidden": DEFAULT_HIDDEN,
            "layers": DEFAULT_LAYERS,
            "dropout": DEFAULT_DROPOUT,
            "nhead": DEFAULT_NHEAD,
            "context_mode": DEFAULT_CONTEXT_MODE,
        }
        argument_help = {
            "hidden": "Transformer hidden dimension.",
            "layers": "Number of Transformer encoder layers.",
            "dropout": "Dropout probability.",
            "nhead": "Number of attention heads.",
            "context_mode": "How NaNs in context timesteps are handled.",
        }
    else:
        defaults = {
            "hidden": None,
            "layers": None,
            "dropout": None,
            "nhead": None,
            "context_mode": None,
        }
        argument_help = {
            "hidden": (
                "Transformer hidden dimension; read from the checkpoint. "
                "If specified, it must match."
            ),
            "layers": (
                "Number of Transformer encoder layers; read from the "
                "checkpoint. If specified, it must match."
            ),
            "dropout": (
                "Dropout probability; read from the checkpoint. If "
                "specified, it must match."
            ),
            "nhead": (
                "Number of attention heads; read from the checkpoint. "
                "If specified, it must match."
            ),
            "context_mode": (
                "NaN handling mode; read from the checkpoint. If specified, "
                "it must match."
            ),
        }

    model.add_argument(
        "--hidden",
        type=positive_int,
        default=defaults["hidden"],
        help=argument_help["hidden"],
    )
    model.add_argument(
        "--layers",
        type=positive_int,
        default=defaults["layers"],
        help=argument_help["layers"],
    )
    model.add_argument(
        "--dropout",
        type=dropout,
        default=defaults["dropout"],
        help=argument_help["dropout"],
    )
    model.add_argument(
        "--nhead",
        type=positive_int,
        default=defaults["nhead"],
        help=argument_help["nhead"],
    )
    model.add_argument(
        "--mode",
        choices=["strict", "relaxed"],
        default=defaults["context_mode"],
        dest="context_mode",
        help=argument_help["context_mode"],
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
        "--loss-stage",
        type=int,
        default=DEFAULT_LOSS_STAGE,
        choices=[4],
        help="Target-aligned objective maximum stage (fixed at 4).",
    )
    train.add_argument(
        "--loss-schedule",
        choices=["none", "epoch", "step"],
        default=DEFAULT_LOSS_SCHEDULE,
        help=(
            "Stage schedule: none starts directly at --loss-stage; epoch/step "
            "advance from stage 1."
        ),
    )
    train.add_argument(
        "--stage-size",
        type=positive_int,
        default=DEFAULT_STAGE_SIZE,
        help="Epoch/step count per loss stage.",
    )
    train.add_argument(
        "--direct-loss-weights",
        type=six_positive_floats,
        default=None,
        metavar="W0,W1,W2,W3,W4,W5",
        help="Positive weights for the six target-aligned direct losses.",
    )
    train.add_argument(
        "--select-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Select the best maximum-stage checkpoint. Disabled by default; "
            "the last maximum-stage checkpoint is then published."
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
        help="Non-improving maximum-stage epochs before early stopping.",
    )
    train.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use mixed precision when running on CUDA.",
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
        runtime.add_argument(
            "--input-spool-dir",
            default=None,
            help=argparse.SUPPRESS,
        )
        runtime.add_argument(
            "--input-frame-count",
            type=nonnegative_int,
            default=None,
            help=argparse.SUPPRESS,
        )
        runtime.add_argument(
            "--recovery-checkpoint-dir",
            default=None,
            help=argparse.SUPPRESS,
        )
        runtime.add_argument(
            "--recovery-events-out",
            default=None,
            help=argparse.SUPPRESS,
        )
        runtime.add_argument(
            "--resume-checkpoint",
            default=None,
            help=argparse.SUPPRESS,
        )
        runtime.add_argument(
            "--recovery-config-hash",
            default=None,
            help=argparse.SUPPRESS,
        )
        runtime.add_argument(
            "--recovery-manifest-hash",
            default=None,
            help=argparse.SUPPRESS,
        )
    _add_model_arguments(parser, required_seq_len=True, training=True)
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
            "Checkpoint input path; relative paths are resolved inside "
            "models/."
        ),
    )
    runtime.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use mixed precision on CUDA; disabled with a diagnostic on CPU.",
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
    _add_model_arguments(parser, required_seq_len=False, training=False)


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
