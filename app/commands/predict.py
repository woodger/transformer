from arrow_io import read_source_arrow, write_arrow
from checkpoint import load_checkpoint_metadata
from data import reshape_source
from factory import build_model, build_trainer
from run_config import model_config_from_args


def run(args, device, build_model_fn=build_model, build_trainer_fn=build_trainer):
    metadata = load_checkpoint_metadata(args.model_name, device)
    model_config = model_config_from_args(
        args,
        checkpoint_config=metadata.get("model_config"),
        require_seq_len=True,
    )

    X_cpu = read_source_arrow(args.data)
    print("X:", X_cpu.shape)
    X_cpu = reshape_source(X_cpu, model_config.seq_len)

    model = build_model_fn(model_config, X_cpu, None, device)
    trainer = build_trainer_fn(args, model, device, model_config)
    trainer.load(args.model_name)
    preds = trainer.predict(X_cpu)
    write_arrow(args.preds_path, preds, args.pred_col)
    print("Predictions saved")
