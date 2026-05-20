import argparse

from config import (
    DEFAULT_DEVICE,
    LR,
    BATCH_SIZE,
    EPOCHS,
    PER_WEEK,
    PATIENCE,
    D_MODEL,
    NUM_LAYERS,
    DROPOUT,
    NHEAD,
)
from version import __version__


APP_DESCRIPTION = """Transformer training and inference CLI.

Modes:
  fit              Train from an Arrow file on disk.
  predict          Load a model and write predictions to an Arrow file.
  fit-stream       Train from framed Arrow payloads on stdin.
  predict-stream   Predict from framed Arrow payloads on stdin.
  plot-metrics     Render SVG charts from a metrics JSONL file in models/.
"""

APP_EPILOG = """Examples:
  python ./app/main.py fit ./data/train.arrow --seq-len=20
  python ./app/main.py fit-stream --seq-len=20 --model-name=model.pth
  python ./app/main.py predict-stream --seq-len=20 --model-name=model.pth
  python ./app/main.py plot-metrics train.jsonl --plots-dir=metrics_plots
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=APP_DESCRIPTION,
        epilog=APP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "action",
        choices=["fit", "predict", "fit-stream", "predict-stream", "plot-metrics"],
        help="Select the execution mode.",
    )

    parser.add_argument(
        "data",
        nargs="?",
        help="Input Arrow file for fit/predict, or metrics JSONL for plot-metrics.",
    )

    runtime = parser.add_argument_group("Runtime")
    model = parser.add_argument_group("Model")
    train = parser.add_argument_group("Training")

    runtime.add_argument("--device", choices=["cpu", "gpu"], default=DEFAULT_DEVICE)
    runtime.add_argument("--model-name", default="model_weights.pth")
    runtime.add_argument("--preds-path", default="/tmp/preds.arrow")
    runtime.add_argument("--pred-col", default="out")
    runtime.add_argument("--metrics-name", default=None)
    runtime.add_argument("--plots-dir", default="metrics_plots")

    model.add_argument("--seq-len", type=int, default=None)
    model.add_argument("--hidden", type=int, default=D_MODEL)
    model.add_argument("--layers", type=int, default=NUM_LAYERS)
    model.add_argument("--dropout", type=float, default=DROPOUT)
    model.add_argument("--nhead", type=int, default=NHEAD)

    train.add_argument("--lr", type=float, default=LR)
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--per-week", type=int, default=PER_WEEK)
    train.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use mixed precision (AMP) if GPU available.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser
