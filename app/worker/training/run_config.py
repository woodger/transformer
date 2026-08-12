from app.config import (
    BATCH_SIZE,
    CONTEXT_MODE,
    D_MODEL,
    DETERMINISTIC,
    DROPOUT,
    EPOCHS,
    LOSS_SCHEDULE,
    LOSS_STAGE,
    LR,
    NHEAD,
    NUM_LAYERS,
    SEED,
    STAGE_SIZE,
    WEIGHT_DECAY,
)
from app.contracts.worker.v3.config import (
    DEFAULT_DIRECT_LOSS_WEIGHTS,
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)


def _pick(value, default):
    return default if value is None else value


def model_config_from_args(
    args,
    checkpoint_config: ModelConfig | dict | None = None,
    require_seq_len: bool = True,
) -> ModelConfig:
    checkpoint_config = _coerce_model_config(checkpoint_config)
    if checkpoint_config is not None:
        _validate_checkpoint_model_overrides(args, checkpoint_config)

    seq_len = _pick(getattr(args, "seq_len", None), _attr(checkpoint_config, "seq_len", None))
    if seq_len is None:
        if require_seq_len:
            raise ValueError("seq_len must be a positive integer")
        seq_len = 0

    config = ModelConfig(
        seq_len=seq_len,
        hidden=_pick(getattr(args, "hidden", None), _attr(checkpoint_config, "hidden", D_MODEL)),
        layers=_pick(getattr(args, "layers", None), _attr(checkpoint_config, "layers", NUM_LAYERS)),
        dropout=_pick(getattr(args, "dropout", None), _attr(checkpoint_config, "dropout", DROPOUT)),
        nhead=_pick(getattr(args, "nhead", None), _attr(checkpoint_config, "nhead", NHEAD)),
        context_mode=_pick(
            getattr(args, "context_mode", None),
            _attr(checkpoint_config, "context_mode", CONTEXT_MODE),
        ),
        out_dim=_pick(getattr(args, "out_dim", None), _attr(checkpoint_config, "out_dim", 6)),
        feature_dim=_attr(checkpoint_config, "feature_dim", None),
    )

    if config.seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    return config


def train_config_from_args(args, checkpoint_config: TrainConfig | dict | None = None) -> TrainConfig:
    checkpoint_config = _coerce_train_config(checkpoint_config)
    selection_requested = getattr(args, "select_best_checkpoint", None)
    selection_min_delta = getattr(args, "selection_min_delta", None)
    selection_patience = getattr(args, "selection_patience", None)
    if (
        selection_requested is not True
        and (selection_min_delta is not None or selection_patience is not None)
    ):
        raise ValueError(
            "selection policy options require --select-best-checkpoint"
        )
    if selection_requested is None:
        selection = _attr(checkpoint_config, "selection", None)
    elif selection_requested:
        selection = CheckpointSelectionConfig(
            min_delta=_pick(selection_min_delta, 0.0),
            patience=_pick(selection_patience, 0),
        )
    else:
        selection = None

    return TrainConfig(
        lr=_pick(getattr(args, "lr", None), _attr(checkpoint_config, "lr", LR)),
        batch_size=_pick(
            getattr(args, "batch_size", None),
            _attr(checkpoint_config, "batch_size", BATCH_SIZE),
        ),
        epochs=_pick(getattr(args, "epochs", None), _attr(checkpoint_config, "epochs", EPOCHS)),
        loss_stage=_pick(
            getattr(args, "loss_stage", None),
            _attr(checkpoint_config, "loss_stage", LOSS_STAGE),
        ),
        loss_schedule=_pick(
            getattr(args, "loss_schedule", None),
            _attr(checkpoint_config, "loss_schedule", LOSS_SCHEDULE),
        ),
        stage_size=_pick(
            getattr(args, "stage_size", None),
            _attr(checkpoint_config, "stage_size", STAGE_SIZE),
        ),
        use_amp=bool(
            _pick(getattr(args, "use_amp", None), _attr(checkpoint_config, "use_amp", False))
        ),
        weight_decay=_pick(
            getattr(args, "weight_decay", None),
            _attr(checkpoint_config, "weight_decay", WEIGHT_DECAY),
        ),
        direct_loss_weights=tuple(
            _pick(
                getattr(args, "direct_loss_weights", None),
                _attr(
                    checkpoint_config,
                    "direct_loss_weights",
                    DEFAULT_DIRECT_LOSS_WEIGHTS,
                ),
            )
        ),
        selection=selection,
        seed=_pick(getattr(args, "seed", None), _attr(checkpoint_config, "seed", SEED)),
        deterministic=bool(
            _pick(
                getattr(args, "deterministic", None),
                _attr(checkpoint_config, "deterministic", DETERMINISTIC),
            )
        ),
    )


def _validate_checkpoint_model_overrides(args, checkpoint_config: ModelConfig):
    options = (
        ("seq_len", "seq-len"),
        ("hidden", "hidden"),
        ("layers", "layers"),
        ("dropout", "dropout"),
        ("nhead", "nhead"),
        ("context_mode", "mode"),
    )
    for attribute, option in options:
        cli_value = getattr(args, attribute, None)
        if cli_value is None:
            continue
        checkpoint_value = getattr(checkpoint_config, attribute)
        if cli_value != checkpoint_value:
            raise ValueError(
                f"--{option}={cli_value} conflicts with checkpoint value "
                f"{checkpoint_value}"
            )


def _attr(obj, name: str, default):
    if obj is None:
        return default
    return getattr(obj, name, default)


def _coerce_model_config(value):
    if isinstance(value, ModelConfig) or value is None:
        return value
    return ModelConfig.from_dict(value)


def _coerce_train_config(value):
    if isinstance(value, TrainConfig) or value is None:
        return value
    return TrainConfig.from_dict(value)
