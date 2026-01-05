import argparse

from config import (
    DEFAULT_DEVICE,
    LR,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    SEQ_LEN,
    D_MODEL,
    NUM_LAYERS,
    DROPOUT,
    NHEAD
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("action", choices=["fit", "predict"])
    parser.add_argument("data")

    parser.add_argument("--device", choices=["cpu", "gpu"], default=DEFAULT_DEVICE)
    parser.add_argument("--model-name", default="model_weights.pth")
    parser.add_argument("--preds-path", default="/tmp/preds.arrow")
    parser.add_argument("--pred-col", default="out")

    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--hidden", type=int, default=D_MODEL)
    parser.add_argument("--layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--nhead", type=int, default=NHEAD)

    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use mixed precision (AMP) if GPU available",
    )

    return parser.parse_args()
