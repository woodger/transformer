import torch

from args import parse_args
from device import get_device
from transformer import TransformerModel
from arrow_io import read_arrow, write_predictions
from trainer import Trainer


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # ---- Data ----
    X_cpu, Y_cpu = read_arrow(args.data)

    total_feat = X_cpu.shape[1]
    if total_feat % args.seq_len != 0:
        raise ValueError("Feature dim not divisible by seq_len")

    feat_dim = total_feat // args.seq_len
    X_cpu = X_cpu.view(X_cpu.size(0), args.seq_len, feat_dim)

    # ---- Model ----
    model = TransformerModel(
        input_dim=feat_dim,
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        out_dim=Y_cpu.shape[1],
        nhead=args.nhead,
    ).to(device)

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    if args.action == "fit":
        trainer.fit(X_cpu, Y_cpu, args.model_path)

    else:  # predict
        trainer.load(args.model_path)
        preds = trainer.predict(X_cpu)
        write_predictions(args.preds_path, preds, args.pred_col)
        print("Predictions saved")


if __name__ == "__main__":
    main()
