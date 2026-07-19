import sys
from dataclasses import replace
from pathlib import Path

from app.data.arrow import (
    DEFAULT_MAX_FRAME_BYTES,
    iter_framed_arrow,
    read_arrow,
    table_to_tensors,
)
from app.data.tensors import reshape_source, validate_feature_dim, validate_target_dim
from app.training.factory import build_model, build_trainer
from app.training.run_config import model_config_from_args


def run(args, device, build_model_fn=build_model, build_trainer_fn=build_trainer):
    input_spool_dir = getattr(args, "input_spool_dir", None)
    input_frame_count = getattr(args, "input_frame_count", None)
    if input_spool_dir is not None or input_frame_count is not None:
        return _run_spooled(
            args,
            device,
            build_model_fn,
            build_trainer_fn,
            input_spool_dir,
            input_frame_count,
        )

    model_config = model_config_from_args(args)
    model = None
    trainer = None
    expected_feat_dim = None
    expected_target_dim = None
    received_frames = 0
    trained_frames = 0
    trained_epochs = 0

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
            print(f"frame {received_frames}, skipped empty payload")
            continue

        X_cpu, Y_cpu = table_to_tensors(table)
        X_cpu = reshape_source(X_cpu, model_config.seq_len)
        expected_feat_dim = validate_feature_dim(X_cpu, expected_feat_dim)
        expected_target_dim = validate_target_dim(Y_cpu, expected_target_dim)

        if model is None:
            model_config = replace(model_config, feature_dim=expected_feat_dim)
            model = build_model_fn(model_config, X_cpu, Y_cpu, device)
            trainer = build_trainer_fn(args, model, device, model_config)
            config_line = getattr(trainer, "config_line", None)
            if config_line is not None:
                print(config_line())
            print("X:", X_cpu.shape, "Y:", Y_cpu.shape)

        def on_epoch(epoch, metrics, monitor_payload):
            nonlocal trained_epochs
            trained_epochs += 1
            print(metrics.console_line(
                frame=received_frames,
                epoch=epoch + 1,
                **getattr(trainer, "metrics_context", {}),
                **monitor_payload,
            ))
            trainer.record_metrics(
                metrics,
                mode="fit-stream",
                frame=received_frames,
                epoch=epoch + 1,
                **monitor_payload,
            )

        trainer.fit_epochs(X_cpu, Y_cpu, on_epoch=on_epoch, frame=received_frames)
        trained_frames += 1

    if trainer is None:
        raise ValueError("No non-empty frames received on stdin")

    trainer.save(args.model_name)
    print(
        f"Model saved after {trained_frames} trained frame(s), "
        f"{trained_epochs} epoch(s) "
        f"from {received_frames} received frame(s)"
    )


def _run_spooled(
    args,
    device,
    build_model_fn,
    build_trainer_fn,
    input_spool_dir,
    input_frame_count,
):
    if input_spool_dir is None or input_frame_count is None:
        raise ValueError(
            "input_spool_dir and input_frame_count must be provided together"
        )
    if type(input_frame_count) is not int or input_frame_count < 0:
        raise ValueError("input_frame_count must be a non-negative integer")

    model_config = model_config_from_args(args)
    model = None
    trainer = None
    expected_feat_dim = None
    expected_target_dim = None
    trained_inputs = []
    input_directory = Path(input_spool_dir)

    # Preflight one durable payload at a time so shape errors are reported
    # before the optimizer mutates model state.
    for ordinal in range(input_frame_count):
        frame = ordinal + 1
        path = input_directory / f"{ordinal}.arrow"
        X_cpu, Y_cpu = read_arrow(str(path))
        if X_cpu.size(0) == 0:
            print(f"frame {frame}, skipped empty payload")
            del X_cpu, Y_cpu
            continue

        X_cpu = reshape_source(X_cpu, model_config.seq_len)
        expected_feat_dim = validate_feature_dim(X_cpu, expected_feat_dim)
        expected_target_dim = validate_target_dim(Y_cpu, expected_target_dim)

        if model is None:
            model_config = replace(model_config, feature_dim=expected_feat_dim)
            model = build_model_fn(model_config, X_cpu, Y_cpu, device)
            trainer = build_trainer_fn(args, model, device, model_config)
            config_line = getattr(trainer, "config_line", None)
            if config_line is not None:
                print(config_line())
            print("X:", X_cpu.shape, "Y:", Y_cpu.shape)

        trained_inputs.append(path)
        del X_cpu, Y_cpu

    if trainer is None:
        raise ValueError("No non-empty frames received in input spool")

    def payloads():
        for path in trained_inputs:
            X_cpu, Y_cpu = read_arrow(str(path))
            X_cpu = reshape_source(X_cpu, model_config.seq_len)
            validate_feature_dim(X_cpu, expected_feat_dim)
            validate_target_dim(Y_cpu, expected_target_dim)
            yield X_cpu, Y_cpu
            del X_cpu, Y_cpu

    trained_epochs = 0

    def on_epoch(epoch, metrics, monitor_payload):
        nonlocal trained_epochs
        trained_epochs += 1
        print(metrics.console_line(
            epoch=epoch + 1,
            **getattr(trainer, "metrics_context", {}),
            **monitor_payload,
        ))
        trainer.record_metrics(
            metrics,
            mode="fit-stream",
            epoch=epoch + 1,
            **monitor_payload,
        )

    trainer.fit_payloads(payloads, on_epoch=on_epoch)
    trainer.save(args.model_name)
    print(
        f"Model saved after {len(trained_inputs)} trained frame(s), "
        f"{trained_epochs} epoch(s) from {input_frame_count} received frame(s)"
    )
