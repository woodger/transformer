import math
from dataclasses import asdict, dataclass

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
    PATIENCE,
    SAVE_BEST_CHECKPOINT,
    SEED,
    STAGE_SIZE,
    TRAIN_MONITOR,
    TRAIN_MONITOR_MIN_IMPROVEMENT,
    WEIGHT_DECAY,
)
from app.training.losses import (
    validate_loss_schedule,
    validate_loss_stage,
    validate_stage_size,
)


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int
    hidden: int = D_MODEL
    layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    nhead: int = NHEAD
    context_mode: str = CONTEXT_MODE
    out_dim: int = 6
    feature_dim: int | None = None

    def __post_init__(self):
        if self.seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if self.hidden <= 0:
            raise ValueError("hidden must be a positive integer")
        if self.layers <= 0:
            raise ValueError("layers must be a positive integer")
        if self.nhead <= 0:
            raise ValueError("nhead must be a positive integer")
        if self.hidden % self.nhead != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by nhead ({self.nhead})"
            )
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.context_mode not in ("strict", "relaxed"):
            raise ValueError("context_mode must be one of: strict, relaxed")
        if self.out_dim != 6:
            raise ValueError("out_dim must be 6")
        if self.feature_dim is not None and self.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None):
        if not data:
            return None

        values = dict(data)
        if "hidden_dim" in values and "hidden" not in values:
            values["hidden"] = values.pop("hidden_dim")
        if "num_layers" in values and "layers" not in values:
            values["layers"] = values.pop("num_layers")

        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class TrainConfig:
    lr: float = LR
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    patience: int = PATIENCE
    loss_stage: int = LOSS_STAGE
    loss_schedule: str = LOSS_SCHEDULE
    stage_size: int = STAGE_SIZE
    use_amp: bool = False
    weight_decay: float = WEIGHT_DECAY
    monitor: str = TRAIN_MONITOR
    monitor_min_improvement: float = TRAIN_MONITOR_MIN_IMPROVEMENT
    save_best_checkpoint: bool = SAVE_BEST_CHECKPOINT
    seed: int = SEED
    deterministic: bool = DETERMINISTIC

    def __post_init__(self):
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be a positive number")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.patience < 0:
            raise ValueError("patience must be a non-negative integer")
        validate_loss_stage(self.loss_stage)
        validate_loss_schedule(self.loss_schedule)
        validate_stage_size(self.stage_size)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative number")
        if self.monitor not in ("loss", "ret_mae", "ret_mae_skill"):
            raise ValueError("monitor must be one of: loss, ret_mae, ret_mae_skill")
        if (
            not math.isfinite(self.monitor_min_improvement)
            or not 0 <= self.monitor_min_improvement < 1
        ):
            raise ValueError("monitor_min_improvement must be in the range [0, 1)")
        if self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None):
        if not data:
            return None

        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


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

    return TrainConfig(
        lr=_pick(getattr(args, "lr", None), _attr(checkpoint_config, "lr", LR)),
        batch_size=_pick(
            getattr(args, "batch_size", None),
            _attr(checkpoint_config, "batch_size", BATCH_SIZE),
        ),
        epochs=_pick(getattr(args, "epochs", None), _attr(checkpoint_config, "epochs", EPOCHS)),
        patience=_pick(
            getattr(args, "patience", None),
            _attr(checkpoint_config, "patience", PATIENCE),
        ),
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
        monitor=_pick(
            getattr(args, "monitor", None),
            _attr(checkpoint_config, "monitor", TRAIN_MONITOR),
        ),
        monitor_min_improvement=_pick(
            getattr(args, "monitor_min_improvement", None),
            _attr(
                checkpoint_config,
                "monitor_min_improvement",
                TRAIN_MONITOR_MIN_IMPROVEMENT,
            ),
        ),
        save_best_checkpoint=bool(
            _pick(
                getattr(args, "save_best_checkpoint", None),
                _attr(checkpoint_config, "save_best_checkpoint", SAVE_BEST_CHECKPOINT),
            )
        ),
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
