import torch
import sys

from args import parse_args
from device import get_device
from transformer import TransformerModel
from arrow_io import iter_framed_arrow, read_arrow, table_to_tensors, write_arrow
from utils import print_stats
from trainer import Trainer


def reshape_source(X_cpu: torch.Tensor, seq_len: int) -> torch.Tensor:
    if seq_len is None or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")

    total_feat = X_cpu.shape[1]
    if total_feat % seq_len != 0:
        raise ValueError("Feature dim not divisible by seq_len")

    feat_dim = total_feat // seq_len

    return X_cpu.view(X_cpu.size(0), seq_len, feat_dim)


def build_model(args, X_cpu: torch.Tensor, Y_cpu: torch.Tensor, device):
    feat_dim = X_cpu.shape[2]

    return TransformerModel(
        input_dim=feat_dim,
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        out_dim=Y_cpu.shape[1],
        nhead=args.nhead
    ).to(device)


def build_trainer(args, model, device):
    return Trainer(
        model=model,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        use_amp=args.use_amp
    )


def fit_stream(args, device):
    model = None
    trainer = None
    received_frames = 0
    trained_frames = 0

    for table in iter_framed_arrow(sys.stdin.buffer):
        received_frames += 1
        if table.num_rows == 0:
            print(f"frame {received_frames}, skipped empty payload")
            continue

        X_cpu, Y_cpu = table_to_tensors(table)
        X_cpu = reshape_source(X_cpu, args.seq_len)

        if model is None:
            model = build_model(args, X_cpu, Y_cpu, device)
            trainer = build_trainer(args, model, device)
            print("X:", X_cpu.shape, "Y:", Y_cpu.shape)

        loss = trainer.fit_batch(X_cpu, Y_cpu, trained_frames)
        trained_frames += 1
        print(f"frame {received_frames}, loss {loss:.6f}")

    if trainer is None:
        raise ValueError("No non-empty frames received on stdin")

    trainer.save(args.model_name)
    print(
        f"Model saved after {trained_frames} trained frame(s) "
        f"from {received_frames} received frame(s)"
    )


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.action == "fit-stream":
        if args.data is not None:
            raise ValueError("fit-stream reads stdin; data path is not supported")
        fit_stream(args, device)
        return

    if args.data is None:
        raise ValueError("data path is required for fit and predict")

    # ---- Data ----
    X_cpu, Y_cpu = read_arrow(args.data)
    print("X:", X_cpu.shape, "Y:", Y_cpu.shape)
    # print_stats(X_cpu)
    
    X_cpu = reshape_source(X_cpu, args.seq_len)

    # ---- Model ----
    model = build_model(args, X_cpu, Y_cpu, device)

    # ---- Trainer ----
    trainer = build_trainer(args, model, device)

    if args.action == "fit":
        trainer.fit(X_cpu, Y_cpu, args.model_name)

    else:  # predict
        trainer.load(args.model_name)
        preds = trainer.predict(X_cpu)
        write_arrow(args.preds_path, preds, args.pred_col)
        print("Predictions saved")


if __name__ == "__main__":
    main()
