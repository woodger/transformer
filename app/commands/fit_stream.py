import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from app.contracts.worker.v3.objective import objective_config_sha256
from app.data.arrow import (
    DEFAULT_MAX_FRAME_BYTES,
    iter_framed_arrow,
    read_arrow,
    table_to_tensors,
)
from app.data.tensors import reshape_source, validate_feature_dim, validate_target_dim
from app.storage.training_recovery import (
    load_training_recovery,
    save_training_recovery,
)
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

        def on_epoch(
            epoch,
            metrics,
            monitor_payload,
            *,
            frame=received_frames,
            active_trainer=trainer,
        ):
            nonlocal trained_epochs
            trained_epochs += 1
            print(metrics.console_line(
                frame=frame,
                epoch=epoch + 1,
                **getattr(active_trainer, "metrics_context", {}),
                **monitor_payload,
            ))
            active_trainer.record_metrics(
                metrics,
                mode="fit-stream",
                frame=frame,
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
    recovery = _recovery_arguments(args)
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

    if recovery is not None and recovery["resume_checkpoint"] is not None:
        try:
            payload = load_training_recovery(
                recovery["resume_checkpoint"],
                device,
                expected_config_hash=recovery["config_hash"],
                expected_manifest_hash=recovery["manifest_hash"],
                expected_objective_config_sha256=objective_config_sha256(
                    trainer.train_config
                ),
            )
            if payload["model_config"] != asdict(model_config):
                raise ValueError(
                    "training recovery model configuration does not match"
                )
            if payload["train_config"] != asdict(trainer.train_config):
                raise ValueError(
                    "training recovery train configuration does not match"
                )
            trainer.load_recovery_state_dict(payload["trainer_state"])
        except Exception as exc:
            raise ValueError(
                "training recovery checkpoint could not be restored"
            ) from exc

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

    def on_epoch_committed(
        _epoch,
        _metrics,
        _monitor_payload,
        _training_complete,
    ):
        generation = trainer.state.global_epoch
        checkpoint_path = os.path.join(
            recovery["checkpoint_dir"],
            f"{generation}.pth",
        )
        event = save_training_recovery(
            checkpoint_path,
            trainer,
            generation=generation,
            config_hash=recovery["config_hash"],
            manifest_hash=recovery["manifest_hash"],
        )
        _append_recovery_event(recovery["events_path"], event)

    if not getattr(trainer, "training_complete", False):
        if recovery is None:
            trainer.fit_payloads(payloads, on_epoch=on_epoch)
        else:
            trainer.fit_payloads_resumable(
                payloads,
                on_epoch=on_epoch,
                on_epoch_committed=on_epoch_committed,
            )
    trainer.save(args.model_name)
    print(
        f"Model saved after {len(trained_inputs)} trained frame(s), "
        f"{trained_epochs} epoch(s) from {input_frame_count} received frame(s)"
    )


def _recovery_arguments(args) -> dict | None:
    values = {
        "checkpoint_dir": getattr(
            args,
            "recovery_checkpoint_dir",
            None,
        ),
        "events_path": getattr(args, "recovery_events_out", None),
        "config_hash": getattr(args, "recovery_config_hash", None),
        "manifest_hash": getattr(args, "recovery_manifest_hash", None),
        "resume_checkpoint": getattr(args, "resume_checkpoint", None),
    }
    required = (
        "checkpoint_dir",
        "events_path",
        "config_hash",
        "manifest_hash",
    )
    configured = [values[name] is not None for name in required]
    if not any(configured):
        if values["resume_checkpoint"] is not None:
            raise ValueError(
                "resume_checkpoint requires training recovery outputs"
            )
        return None
    if not all(configured):
        raise ValueError(
            "training recovery arguments must be provided together"
        )
    return values


def _append_recovery_event(path: str, event: dict) -> None:
    destination = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    line = json.dumps(
        event,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    with open(destination, "ab", buffering=0) as target:
        target.write(line)
        os.fsync(target.fileno())
