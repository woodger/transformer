from dataclasses import replace

from app.data.arrow import read_arrow
from app.data.tensors import reshape_source
from app.training.factory import build_model, build_trainer
from app.training.run_config import model_config_from_args


def run(args, device, build_model_fn=build_model, build_trainer_fn=build_trainer):
    model_config = model_config_from_args(args)
    X_cpu, Y_cpu = read_arrow(args.data)
    print("X:", X_cpu.shape, "Y:", Y_cpu.shape)
    if X_cpu.shape[0] == 0:
        raise ValueError("Training input contains no rows")
    X_cpu = reshape_source(X_cpu, model_config.seq_len)
    model_config = replace(model_config, feature_dim=X_cpu.shape[2])

    model = build_model_fn(model_config, X_cpu, Y_cpu, device)
    trainer = build_trainer_fn(args, model, device, model_config)
    trainer.fit(X_cpu, Y_cpu, args.model_name)
