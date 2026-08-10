import argparse
import math

from app.config import (
    BATCH_SIZE,
    CONTEXT_MODE,
    D_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FRAME_BYTES,
    DETERMINISTIC,
    DROPOUT,
    EPOCHS,
    HOST_DEFAULT,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    LR,
    NHEAD,
    NUM_LAYERS,
    PATIENCE,
    PORT_DEFAULT,
    SAVE_BEST_CHECKPOINT,
    SEED,
    STAGE_SIZE,
    TRAIN_MONITOR,
    TRAIN_MONITOR_MIN_IMPROVEMENT,
    WEIGHT_DECAY,
)
from app.version import __version__, version_text

_COMMAND_GROUPS = (
    (
        "Flight",
        (("flight serve", "Run the durable Arrow Flight job service."),),
    ),
    (
        "Access",
        (
            ("auth tokens issue", "Issue a local API access token"),
            ("auth tokens list", "List API access token metadata"),
            ("auth tokens revoke <token-id>", "Revoke an API access token"),
        ),
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
        "Diagnostics",
        (("gmark", "Stress one CUDA GPU and verify compute integrity."),),
    ),
    (
        "Metrics",
        (("plot-metrics", "Render SVG charts from metrics JSONL."),),
    ),
    (
        "Database",
        (
            ("db migrations status", "Read-only schema migration state"),
            ("db migrations apply", "Apply pending schema migrations"),
            ("db migrations rollback", "Revert the latest schema migration"),
        ),
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
    "gmark": """Examples:
  transformer gmark --duration=300 --memory-fraction=0.7
""",
}

_FLIGHT_SERVE_DESCRIPTION = (
    "Run the durable Arrow Flight service for fit and predict jobs."
)


def _format_root_help() -> str:
    command_groups = "\n\n".join(
        f"{group}:\n"
        + "\n".join(
            f"  {name:<32}{description}" for name, description in commands
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
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


class _FlightServiceHelpFormatter(argparse.RawTextHelpFormatter):
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


def _add_hidden_help_argument(parser):
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help=argparse.SUPPRESS,
    )


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


def _gmark_memory_fraction(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed > 0.9:
        raise argparse.ArgumentTypeError("must be in the range [0, 0.9]")
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
            help=(
                "Sequence length; read from the checkpoint, but required for "
                "legacy checkpoints."
            ),
        )

    if training:
        defaults = {
            "hidden": D_MODEL,
            "layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "nhead": NHEAD,
            "context_mode": CONTEXT_MODE,
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
                "Transformer hidden dimension; read from the checkpoint, "
                f"or {D_MODEL} for legacy checkpoints."
            ),
            "layers": (
                "Number of Transformer encoder layers; read from the checkpoint, "
                f"or {NUM_LAYERS} for legacy checkpoints."
            ),
            "dropout": (
                "Dropout probability; read from the checkpoint, "
                f"or {DROPOUT} for legacy checkpoints."
            ),
            "nhead": (
                "Number of attention heads; read from the checkpoint, "
                f"or {NHEAD} for legacy checkpoints."
            ),
            "context_mode": (
                "NaN handling mode; read from the checkpoint, "
                f"or {CONTEXT_MODE} for legacy checkpoints."
            ),
        }

    model.add_argument(
        "--hidden",
        type=_positive_int,
        default=defaults["hidden"],
        help=argument_help["hidden"],
    )
    model.add_argument(
        "--layers",
        type=_positive_int,
        default=defaults["layers"],
        help=argument_help["layers"],
    )
    model.add_argument(
        "--dropout",
        type=_dropout,
        default=defaults["dropout"],
        help=argument_help["dropout"],
    )
    model.add_argument(
        "--nhead",
        type=_positive_int,
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
        add_help=False,
        help=_COMMAND_HELP[name],
        epilog=_COMMAND_EXAMPLES.get(name),
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(parser)
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
            type=_nonnegative_int,
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


def _add_predict_parser(subparsers, name: str, *, stream: bool):
    parser = subparsers.add_parser(
        name,
        add_help=False,
        help=_COMMAND_HELP[name],
        epilog=_COMMAND_EXAMPLES.get(name),
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(parser)
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


def _add_gmark_parser(subparsers):
    parser = subparsers.add_parser(
        "gmark",
        add_help=False,
        help=_COMMAND_HELP["gmark"],
        description=(
            "Stress one CUDA GPU with repeated matrix multiplications and an "
            "optional active VRAM allocation. Press Ctrl+C to stop."
        ),
        epilog=_COMMAND_EXAMPLES["gmark"],
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(parser)
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=60.0,
        metavar="SECONDS",
        help="Test duration after warm-up.",
    )
    parser.add_argument(
        "--device",
        type=_nonnegative_int,
        default=0,
        metavar="INDEX",
        help="Logical CUDA device index.",
    )
    parser.add_argument(
        "--matrix-size",
        type=_positive_int,
        default=8192,
        metavar="N",
        help="Multiply square N x N matrices.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Matrix element type.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=_gmark_memory_fraction,
        default=0.7,
        metavar="FRACTION",
        help=(
            "Fraction of VRAM free after warm-up to reserve and write; "
            "0 disables the ballast."
        ),
    )
    parser.add_argument(
        "--max-temperature",
        type=_nonnegative_float,
        default=80.0,
        metavar="CELSIUS",
        help=(
            "Stop at this GPU temperature; nvidia-smi is required unless "
            "0 disables the cutoff."
        ),
    )
    parser.add_argument(
        "--status-interval",
        type=_positive_float,
        default=2.0,
        metavar="SECONDS",
        help="Status and integrity-check interval.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=_positive_int,
        default=3,
        metavar="COUNT",
        help="Matrix multiplications before timing starts.",
    )
    parser.add_argument(
        "--sync-every",
        type=_positive_int,
        default=4,
        metavar="COUNT",
        help="Synchronize CUDA after this many multiplications.",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=12345,
        help="Random seed for input matrices.",
    )
    parser.set_defaults(data=None, metrics_name=None)


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
    _add_gmark_parser(subparsers)

    flight = subparsers.add_parser(
        "flight",
        help="Arrow Flight service commands.",
        formatter_class=_HelpFormatter,
    )
    flight_commands = flight.add_subparsers(
        parser_class=_ArgumentParser,
        dest="flight_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    service = flight_commands.add_parser(
        "serve",
        add_help=False,
        help=_COMMAND_HELP["flight serve"],
        description=_FLIGHT_SERVE_DESCRIPTION,
        formatter_class=_FlightServiceHelpFormatter,
        usage="%(prog)s [options]",
    )
    _add_hidden_help_argument(service)
    service.add_argument(
        "--host",
        metavar="HOST",
        default=None,
        help=f"Listen host. (default: {HOST_DEFAULT})",
    )
    service.add_argument(
        "--port",
        type=_nonnegative_int,
        default=None,
        help=f"Listen port. (default: {PORT_DEFAULT})",
    )

    service.add_argument(
        "--allow-plaintext",
        action="store_true",
        default=None,
        help="Allow serving without TLS.",
    )

    service.add_argument(
        "--tls-cert-file",
        default=None,
        metavar="FILE",
        help="Server certificate PEM.\nRequires --tls-key-file.",
    )
    service.add_argument(
        "--tls-key-file",
        default=None,
        metavar="FILE",
        help="Server private key PEM.\nRequires --tls-cert-file.",
    )

    service.add_argument(
        "--tls-ca-file",
        default=None,
        metavar="FILE",
        help="Client CA PEM.\nRequires server TLS.",
    )
    service.add_argument(
        "--tls-require-client-cert",
        action="store_true",
        default=None,
        help=(
            "Require client certificates.\n"
            "Requires --tls-ca-file and server TLS."
        ),
    )

    service.set_defaults(data=None, metrics_name=None)

    auth = subparsers.add_parser(
        "auth",
        help="API access commands.",
        formatter_class=_HelpFormatter,
    )
    auth_commands = auth.add_subparsers(
        parser_class=_ArgumentParser,
        dest="auth_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    tokens = auth_commands.add_parser(
        "tokens",
        help="API access token commands.",
        formatter_class=_HelpFormatter,
    )
    token_commands = tokens.add_subparsers(
        parser_class=_ArgumentParser,
        dest="tokens_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    issue = token_commands.add_parser(
        "issue",
        add_help=False,
        description="Issue an API access token and print its credential.",
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(issue)
    issue.add_argument(
        "--subject",
        required=True,
        help="Authenticated subject associated with the token.",
    )
    issue.set_defaults(data=None, metrics_name=None)
    token_list = token_commands.add_parser(
        "list",
        add_help=False,
        description="List API access token metadata without credentials.",
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(token_list)
    token_list.set_defaults(data=None, metrics_name=None)
    revoke = token_commands.add_parser(
        "revoke",
        add_help=False,
        description="Revoke an API access token by ID.",
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(revoke)
    revoke.add_argument("token_id", metavar="TOKEN_ID", help="API access token UUID.")
    revoke.set_defaults(data=None, metrics_name=None)

    database = subparsers.add_parser(
        "db",
        help="Database schema commands.",
        formatter_class=_HelpFormatter,
    )
    database_commands = database.add_subparsers(
        parser_class=_ArgumentParser,
        dest="db_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    migrations = database_commands.add_parser(
        "migrations",
        help="Database schema migration commands.",
        formatter_class=_HelpFormatter,
    )
    migration_commands = migrations.add_subparsers(
        parser_class=_ArgumentParser,
        dest="migrations_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    for name, description in (
        ("status", "Read the current and expected schema revisions."),
        ("apply", "Apply all pending schema migrations."),
        ("rollback", "Revert the latest applied schema migration."),
    ):
        command = migration_commands.add_parser(
            name,
            add_help=False,
            description=description,
            formatter_class=_HelpFormatter,
        )
        _add_hidden_help_argument(command)
        command.set_defaults(data=None, metrics_name=None)

    plot = subparsers.add_parser(
        "plot-metrics",
        add_help=False,
        help=_COMMAND_HELP["plot-metrics"],
        formatter_class=_HelpFormatter,
    )
    _add_hidden_help_argument(plot)
    plot.add_argument("data", metavar="METRICS_FILE", help="Metrics JSONL file.")
    plot.add_argument(
        "--plots-dir",
        default="metrics_plots",
        help="Directory for generated SVG charts.",
    )
    plot.set_defaults(metrics_name=None)

    return parser
