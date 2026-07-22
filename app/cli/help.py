import argparse
import math

from app.config import (
    BATCH_SIZE,
    CONTEXT_MODE,
    DEFAULT_DEVICE,
    DETERMINISTIC,
    D_MODEL,
    DROPOUT,
    EPOCHS,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    LR,
    NHEAD,
    NUM_LAYERS,
    PATIENCE,
    SEED,
    SAVE_BEST_CHECKPOINT,
    STAGE_SIZE,
    TRAIN_MONITOR,
    TRAIN_MONITOR_MIN_IMPROVEMENT,
    WEIGHT_DECAY,
)
from app.data.arrow import DEFAULT_MAX_FRAME_BYTES
from app.runtime.version import __version__, version_text


_COMMAND_GROUPS = (
    (
        "Server",
        (("serve-flight", "Run the durable Arrow Flight job service."),),
    ),
    (
        "Training and inference",
        (
            ("fit", "Train from an Arrow file."),
            ("predict", "Predict from an Arrow file."),
            ("fit-stream", "Train from framed stdin."),
            ("predict-stream", "Predict from framed stdin."),
        ),
    ),
    (
        "Metrics",
        (("plot-metrics", "Render SVG charts from metrics JSONL."),),
    ),
)
_COMMAND_HELP = {
    name: description
    for _, commands in _COMMAND_GROUPS
    for name, description in commands
}
_COMMAND_EXAMPLES = {
    "fit": """Examples:
  transformer fit ./data/train.arrow --seq-len=20
""",
    "predict": """Examples:
  transformer predict ./data/test.arrow --checkpoint=model.pth --output=/tmp/preds.arrow
""",
    "serve-flight": """Examples:
  transformer serve-flight --config=/etc/transformer/flight.json
""",
}


def _format_root_help() -> str:
    command_groups = "\n\n".join(
        f"{group}:\n"
        + "\n".join(
            f"  {name:<16}{description}" for name, description in commands
        )
        for group, commands in _COMMAND_GROUPS
    )
    return (
        f"transformer {__version__}\n\n"
        "Usage:\n"
        "  transformer <command> [args] [options]\n"
        "  transformer <command> --help\n"
        "  transformer --help\n"
        "  transformer --version\n\n"
        "Global options:\n"
        "  --help, -h       Show help and exit\n"
        "  --version, -v    Show package and runtime version info\n\n"
        "Commands:\n\n"
        f"{command_groups}\n\n"
        "Command details:\n"
        "  transformer <command> --help\n"
    )


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        hidden = getattr(parsed, "hidden", None)
        nhead = getattr(parsed, "nhead", None)
        if hidden is not None and nhead is not None and hidden % nhead != 0:
            self.error(
                f"--hidden={hidden} must be divisible by --nhead={nhead}"
            )
        return parsed


class _RootArgumentParser(_ArgumentParser):
    def format_help(self):
        return _format_root_help()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _seed(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed > 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def _dropout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)") from exc
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)")
    return parsed


def _fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)") from exc
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)")
    return parsed


def _add_device_argument(group):
    group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_DEVICE,
        help="Runtime device. auto uses CUDA when available.",
    )


def _add_max_frame_bytes_argument(group):
    group.add_argument(
        "--max-frame-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_FRAME_BYTES,
        help="Maximum accepted framed Arrow payload size in bytes.",
    )


def _add_model_arguments(parser, *, required_seq_len: bool, training: bool):
    model = parser.add_argument_group("Model")
    if required_seq_len:
        model.add_argument(
            "--seq-len",
            type=_positive_int,
            required=True,
            default=argparse.SUPPRESS,
            help="Sequence length.",
        )
    else:
        model.add_argument(
            "--seq-len",
            type=_positive_int,
            default=None,
            help="Legacy override; otherwise read from the checkpoint.",
        )

    defaults = {
        "hidden": D_MODEL,
        "layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "nhead": NHEAD,
        "context_mode": CONTEXT_MODE,
    } if training else {
        "hidden": None,
        "layers": None,
        "dropout": None,
        "nhead": None,
        "context_mode": None,
    }

    model.add_argument(
        "--hidden",
        type=_positive_int,
        default=defaults["hidden"],
        help="Transformer hidden dimension; prediction may read it from checkpoint.",
    )
    model.add_argument(
        "--layers",
        type=_positive_int,
        default=defaults["layers"],
        help="Number of Transformer encoder layers; prediction may read it from checkpoint.",
    )
    model.add_argument(
        "--dropout",
        type=_dropout,
        default=defaults["dropout"],
        help="Dropout probability; prediction may read it from checkpoint.",
    )
    model.add_argument(
        "--nhead",
        type=_positive_int,
        default=defaults["nhead"],
        help="Number of attention heads; prediction may read it from checkpoint.",
    )
    model.add_argument(
        "--mode",
        choices=["strict", "relaxed"],
        default=defaults["context_mode"],
        dest="context_mode",
        help="How NaNs in context timesteps are handled.",
    )


def _add_training_arguments(parser):
    train = parser.add_argument_group("Training")
    train.add_argument("--lr", type=_positive_float, default=LR, help="Learning rate.")
    train.add_argument(
        "--weight-decay",
        type=_nonnegative_float,
        default=WEIGHT_DECAY,
        help="Adam weight decay.",
    )
    train.add_argument(
        "--batch-size",
        type=_positive_int,
        default=BATCH_SIZE,
        help="Mini-batch size.",
    )
    train.add_argument(
        "--epochs",
        type=_positive_int,
        default=EPOCHS,
        help="Epoch count.",
    )
    train.add_argument(
        "--loss-stage",
        type=int,
        default=LOSS_STAGE,
        choices=[1, 2, 3, 4],
        help=(
            "Maximum loss stage: 1=return, 2=+probabilities, "
            "3=+Bayesian EV, 4=+volatility."
        ),
    )
    train.add_argument(
        "--loss-schedule",
        choices=["none", "epoch", "step"],
        default=LOSS_SCHEDULE,
        help=(
            "Stage schedule: none starts directly at --loss-stage; epoch/step "
            "advance from stage 1."
        ),
    )
    train.add_argument(
        "--stage-size",
        type=_positive_int,
        default=STAGE_SIZE,
        help="Epoch/step count per loss stage.",
    )
    train.add_argument(
        "--patience",
        type=_nonnegative_int,
        default=PATIENCE,
        help=(
            "Non-improving training epochs allowed at the final active loss "
            "stage; 0 disables early stopping."
        ),
    )
    train.add_argument(
        "--monitor",
        choices=["loss", "ret_mae", "ret_mae_skill"],
        default=TRAIN_MONITOR,
        help="Training-pass metric used for early stopping and checkpoint selection.",
    )
    train.add_argument(
        "--monitor-min-improvement",
        type=_fraction,
        default=TRAIN_MONITOR_MIN_IMPROVEMENT,
        help="For MAE monitors, required fractional improvement over the zero-return baseline.",
    )
    train.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use mixed precision when running on CUDA.",
    )
    train.add_argument(
        "--seed",
        type=_seed,
        default=SEED,
        help="Random seed applied to Python, NumPy and PyTorch and saved in config.",
    )
    train.add_argument(
        "--deterministic",
        action="store_true",
        default=DETERMINISTIC,
        help="Request deterministic PyTorch algorithms (unsupported ops may fail).",
    )
    train.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=SAVE_BEST_CHECKPOINT,
        help="Publish the best monitored checkpoint instead of the final weights.",
    )


def _add_fit_parser(subparsers, name: str, *, stream: bool):
    parser = subparsers.add_parser(
        name,
        help=_COMMAND_HELP[name],
        epilog=_COMMAND_EXAMPLES.get(name),
        formatter_class=_HelpFormatter,
    )
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
        help="Checkpoint output path; relative paths are resolved inside models/.",
    )
    runtime.add_argument(
        "--metrics-out",
        "--metrics-name",
        dest="metrics_name",
        default=None,
        help="Metrics JSONL output path; relative paths are resolved inside models/.",
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
            type=_nonnegative_int,
            default=None,
            help=argparse.SUPPRESS,
        )
    _add_model_arguments(parser, required_seq_len=True, training=True)
    _add_training_arguments(parser)


def _add_predict_parser(subparsers, name: str, *, stream: bool):
    parser = subparsers.add_parser(
        name,
        help=_COMMAND_HELP[name],
        epilog=_COMMAND_EXAMPLES.get(name),
        formatter_class=_HelpFormatter,
    )
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
        help="Checkpoint input path; relative paths are resolved inside models/.",
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


def build_parser():
    parser = _RootArgumentParser(
        prog="transformer",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=version_text("%(prog)s"),
    )

    subparsers = parser.add_subparsers(
        parser_class=_ArgumentParser,
        dest="action",
        required=True,
        title="commands",
        metavar="COMMAND",
    )
    _add_fit_parser(subparsers, "fit", stream=False)
    _add_predict_parser(subparsers, "predict", stream=False)
    _add_fit_parser(subparsers, "fit-stream", stream=True)
    _add_predict_parser(subparsers, "predict-stream", stream=True)

    service = subparsers.add_parser(
        "serve-flight",
        help=_COMMAND_HELP["serve-flight"],
        epilog=_COMMAND_EXAMPLES["serve-flight"],
        formatter_class=_HelpFormatter,
    )
    service.add_argument(
        "--config",
        default=None,
        help="JSON service configuration file; environment and CLI override it.",
    )
    service.add_argument("--state-dir", default=None, help="Persistent service state directory.")
    service.add_argument(
        "--host",
        metavar="HOST",
        default=None,
        help="Flight listen host.",
    )
    service.add_argument("--port", type=_nonnegative_int, default=None, help="Flight port.")
    service.add_argument(
        "--profile",
        choices=["production", "development", "lan"],
        default=None,
        help="Security profile.",
    )
    service.add_argument(
        "--allow-plaintext",
        action="store_true",
        default=None,
        help="Explicitly allow plaintext in development/LAN profile.",
    )
    service.add_argument("--tls-cert-file", default=None, help="TLS certificate PEM file.")
    service.add_argument("--tls-key-file", default=None, help="TLS private key PEM file.")
    service.add_argument("--tls-ca-file", default=None, help="mTLS client CA PEM file.")
    service.add_argument(
        "--tls-require-client-cert",
        action="store_true",
        default=None,
        help="Require and verify client certificates.",
    )
    service.add_argument(
        "--bearer-tokens-file",
        default=None,
        help="Secret JSON token-to-subject mapping file.",
    )
    service.set_defaults(data=None, metrics_name=None)

    plot = subparsers.add_parser(
        "plot-metrics",
        help=_COMMAND_HELP["plot-metrics"],
        formatter_class=_HelpFormatter,
    )
    plot.add_argument("data", metavar="METRICS_FILE", help="Metrics JSONL file.")
    plot.add_argument(
        "--plots-dir",
        default="metrics_plots",
        help="Directory for generated SVG charts.",
    )
    plot.set_defaults(metrics_name=None)

    return parser
