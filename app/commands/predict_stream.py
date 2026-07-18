import sys
from contextlib import redirect_stdout

from app.data.arrow import (
    DEFAULT_MAX_FRAME_BYTES,
    empty_predictions_table,
    iter_framed_arrow,
    predictions_to_table,
    table_to_source_tensor,
    write_framed_arrow,
)
from app.storage.checkpoint import load_checkpoint_metadata
from app.data.tensors import (
    reshape_source,
    validate_checkpoint_feature_dim,
    validate_feature_dim,
)
from app.training.factory import build_model, build_trainer
from app.training.run_config import model_config_from_args


def run(args, device, build_model_fn=build_model, build_trainer_fn=build_trainer):
    model_config = None
    model = None
    trainer = None
    expected_feat_dim = None
    received_frames = 0
    predicted_frames = 0

    for table in iter_framed_arrow(
        sys.stdin.buffer,
        max_frame_bytes=getattr(
            args,
            "max_frame_bytes",
            DEFAULT_MAX_FRAME_BYTES,
        ),
    ):
        received_frames += 1
        if table.num_rows == 0:
            write_framed_arrow(
                sys.stdout.buffer,
                empty_predictions_table(args.pred_col),
            )
            print(
                f"frame {received_frames}, emitted empty predictions",
                file=sys.stderr,
            )
            continue

        if model_config is None:
            with redirect_stdout(sys.stderr):
                checkpoint_config = _checkpoint_model_config(args, device)
            model_config = model_config_from_args(
                args,
                checkpoint_config=checkpoint_config,
                require_seq_len=True,
            )

        X_cpu = table_to_source_tensor(table)
        X_cpu = reshape_source(X_cpu, model_config.seq_len)
        validate_checkpoint_feature_dim(X_cpu, model_config.feature_dim)
        expected_feat_dim = validate_feature_dim(X_cpu, expected_feat_dim)

        if model is None:
            with redirect_stdout(sys.stderr):
                model = build_model_fn(model_config, X_cpu, None, device)
                trainer = build_trainer_fn(args, model, device, model_config)
                trainer.load(args.model_name)
            print("X:", X_cpu.shape, file=sys.stderr)
            print("Model loaded", file=sys.stderr)

        with redirect_stdout(sys.stderr):
            preds = trainer.predict(X_cpu)
        write_framed_arrow(
            sys.stdout.buffer,
            predictions_to_table(
                preds,
                args.pred_col,
                expected_rows=X_cpu.shape[0],
            ),
        )
        predicted_frames += 1
        print(
            f"frame {received_frames}, predicted {preds.shape[0]} row(s)",
            file=sys.stderr,
        )

    print(
        f"Predicted {predicted_frames} non-empty frame(s) "
        f"from {received_frames} received frame(s)",
        file=sys.stderr,
    )


def _checkpoint_model_config(args, device):
    try:
        metadata = load_checkpoint_metadata(args.model_name, device)
    except FileNotFoundError:
        return None
    return metadata.get("model_config")
