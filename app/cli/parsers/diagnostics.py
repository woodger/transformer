from app.cli.formatting import COMMAND_EXAMPLES, COMMAND_HELP, HelpFormatter
from app.cli.options import (
    add_hidden_help_argument,
    gmark_memory_fraction,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    seed,
)
from app.cli.parsers import SubparserTarget
from app.contracts.worker.v4.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_NHEAD,
    DEFAULT_SEED,
)


def add_diagnostic_parsers(subparsers: SubparserTarget) -> None:
    parser = subparsers.add_parser(
        "gmark",
        add_help=False,
        help=COMMAND_HELP["gmark"],
        description=(
            "Stress one CUDA GPU with synthetic Transformer training and an "
            "optional active VRAM allocation. Press Ctrl+C to stop."
        ),
        epilog=COMMAND_EXAMPLES["gmark"],
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(parser)
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=60.0,
        metavar="SECONDS",
        help="Test duration after warm-up.",
    )
    parser.add_argument(
        "--device",
        type=nonnegative_int,
        default=0,
        metavar="INDEX",
        help="Logical CUDA device index.",
    )
    parser.add_argument(
        "--seq-len",
        type=positive_int,
        default=10,
        metavar="LENGTH",
        help="Input sequence length.",
    )
    parser.add_argument(
        "--feature-dim",
        type=positive_int,
        default=891,
        metavar="COUNT",
        help="Features per input timestep.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        metavar="COUNT",
        help="Rows per optimizer step.",
    )
    parser.add_argument(
        "--hidden",
        type=positive_int,
        default=DEFAULT_HIDDEN,
        metavar="SIZE",
        help="Transformer hidden dimension.",
    )
    parser.add_argument(
        "--layers",
        type=positive_int,
        default=DEFAULT_LAYERS,
        metavar="COUNT",
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument(
        "--nhead",
        type=positive_int,
        default=DEFAULT_NHEAD,
        metavar="COUNT",
        help="Number of attention heads; --hidden must be divisible by it.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use the production CUDA autocast and GradScaler path.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=gmark_memory_fraction,
        default=0.0,
        metavar="FRACTION",
        help=(
            "Fraction of VRAM free after warm-up to reserve and write; "
            "0 disables the ballast."
        ),
    )
    parser.add_argument(
        "--max-temperature",
        type=nonnegative_float,
        default=80.0,
        metavar="CELSIUS",
        help=(
            "Stop at this GPU temperature; nvidia-smi is required unless "
            "0 disables the cutoff."
        ),
    )
    parser.add_argument(
        "--status-interval",
        type=positive_float,
        default=2.0,
        metavar="SECONDS",
        help="Status and integrity-check interval.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=positive_int,
        default=3,
        metavar="COUNT",
        help="Optimizer steps before timing starts.",
    )
    parser.add_argument(
        "--seed",
        type=seed,
        default=DEFAULT_SEED,
        help="Random seed for model parameters and synthetic data.",
    )
    parser.set_defaults(data=None, metrics_name=None)


__all__ = ["add_diagnostic_parsers"]
