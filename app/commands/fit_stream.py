import sys

from app.data.arrow import iter_framed_arrow, table_to_tensors
from app.data.tensors import reshape_source, validate_feature_dim, validate_target_dim
from app.training.factory import build_model, build_trainer
from app.training.run_config import model_config_from_args


def run(args, device, build_model_fn=build_model, build_trainer_fn=build_trainer):
    model_config = model_config_from_args(args)
    model = None
    trainer = None
    expected_feat_dim = None
    expected_target_dim = None
    received_frames = 0
    trained_frames = 0
    trained_epochs = 0

    for table in iter_framed_arrow(sys.stdin.buffer):
        received_frames += 1
        if table.num_rows == 0:
            print(f"frame {received_frames}, skipped empty payload")
            continue

        X_cpu, Y_cpu = table_to_tensors(table)
        X_cpu = reshape_source(X_cpu, model_config.seq_len)
        expected_feat_dim = validate_feature_dim(X_cpu, expected_feat_dim)
        expected_target_dim = validate_target_dim(Y_cpu, expected_target_dim)

        if model is None:
            model = build_model_fn(model_config, X_cpu, Y_cpu, device)
            trainer = build_trainer_fn(args, model, device, model_config)
            print("X:", X_cpu.shape, "Y:", Y_cpu.shape)

        def on_epoch(epoch, metrics):
            nonlocal trained_epochs
            trained_epochs += 1
            print(metrics.log_line(
                frame=received_frames,
                epoch=epoch + 1,
                **getattr(trainer, "metrics_context", {}),
            ))
            trainer.record_metrics(
                metrics,
                mode="fit-stream",
                frame=received_frames,
                epoch=epoch + 1,
            )

        trainer.fit_epochs(X_cpu, Y_cpu, on_epoch=on_epoch)
        trained_frames += 1

    if trainer is None:
        raise ValueError("No non-empty frames received on stdin")

    trainer.save(args.model_name)
    print(
        f"Model saved after {trained_frames} trained frame(s), "
        f"{trained_epochs} epoch(s) "
        f"from {received_frames} received frame(s)"
    )
