import os

import torch

from app.data.arrow import read_source_arrow, write_arrow
from app.storage.checkpoint import load_checkpoint_metadata
from app.data.tensors import reshape_source, validate_checkpoint_feature_dim
from app.training.factory import build_model, build_trainer
from app.training.run_config import model_config_from_args


def run(args, device, build_model_fn=build_model, build_trainer_fn=build_trainer):
    if os.path.realpath(args.data) == os.path.realpath(args.preds_path):
        raise ValueError("prediction output path must differ from input data path")

    metadata = load_checkpoint_metadata(args.model_name, device)
    model_config = model_config_from_args(
        args,
        checkpoint_config=metadata.get("model_config"),
        require_seq_len=True,
    )

    X_cpu = read_source_arrow(args.data)
    print("X:", X_cpu.shape)
    if X_cpu.shape[0] == 0:
        write_arrow(
            args.preds_path,
            torch.empty((0, model_config.out_dim), dtype=torch.float32),
            args.pred_col,
        )
        print("Predictions saved")
        return
    X_cpu = reshape_source(X_cpu, model_config.seq_len)
    validate_checkpoint_feature_dim(X_cpu, model_config.feature_dim)

    model = build_model_fn(model_config, X_cpu, None, device)
    trainer = build_trainer_fn(args, model, device, model_config)
    trainer.load(args.model_name)
    preds = trainer.predict(X_cpu)
    write_arrow(
        args.preds_path,
        preds,
        args.pred_col,
        expected_rows=X_cpu.shape[0],
    )
    print("Predictions saved")
