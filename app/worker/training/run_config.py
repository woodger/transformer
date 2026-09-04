from collections.abc import Sequence
from typing import TypeVar, cast

from app.contracts.ml import MAX_TARGET_WIDTH
from app.contracts.worker.v11.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_DETERMINISTIC,
    DEFAULT_DIRECT_LOSS_WEIGHTS,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_LR,
    DEFAULT_NHEAD,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    CheckpointSelectionConfig,
    ModelConfig,
    TrainConfig,
)
from app.contracts.worker.v11.objective import ObjectiveConfig, default_objective

T = TypeVar("T")


def _pick(value: T | None, default: T) -> T:
    return default if value is None else value


def model_config_from_args(
    args: object,
    checkpoint_config: object = None,
    require_seq_len: bool = True,
) -> ModelConfig:
    checkpoint_config = _coerce_model_config(checkpoint_config)
    if checkpoint_config is not None:
        _validate_checkpoint_model_overrides(args, checkpoint_config)

    seq_len = _pick(
        _optional_arg(args, "seq_len", int),
        None if checkpoint_config is None else checkpoint_config.seq_len,
    )
    if seq_len is None:
        if require_seq_len:
            raise ValueError("seq_len must be a positive integer")
        seq_len = 0

    config = ModelConfig(
        seq_len=seq_len,
        hidden=_pick(
            _optional_arg(args, "hidden", int),
            (
                DEFAULT_HIDDEN
                if checkpoint_config is None
                else checkpoint_config.hidden
            ),
        ),
        layers=_pick(
            _optional_arg(args, "layers", int),
            (
                DEFAULT_LAYERS
                if checkpoint_config is None
                else checkpoint_config.layers
            ),
        ),
        dropout=_pick(
            _optional_arg(args, "dropout", float),
            (
                DEFAULT_DROPOUT
                if checkpoint_config is None
                else checkpoint_config.dropout
            ),
        ),
        nhead=_pick(
            _optional_arg(args, "nhead", int),
            (
                DEFAULT_NHEAD
                if checkpoint_config is None
                else checkpoint_config.nhead
            ),
        ),
        context_mode=_pick(
            _optional_arg(args, "context_mode", str),
            (
                DEFAULT_CONTEXT_MODE
                if checkpoint_config is None
                else checkpoint_config.context_mode
            ),
        ),
        out_dim=_pick(
            _optional_arg(args, "out_dim", int),
            (
                MAX_TARGET_WIDTH
                if checkpoint_config is None
                else checkpoint_config.out_dim
            ),
        ),
        feature_dim=(
            None if checkpoint_config is None else checkpoint_config.feature_dim
        ),
    )

    if config.seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    return config


def train_config_from_args(
    args: object,
    checkpoint_config: object = None,
) -> TrainConfig:
    checkpoint_config = _coerce_train_config(checkpoint_config)
    selection_requested = _optional_arg(args, "select_best_checkpoint", bool)
    selection_min_delta = _optional_arg(args, "selection_min_delta", float)
    selection_patience = _optional_arg(args, "selection_patience", int)
    if (
        selection_requested is not True
        and (selection_min_delta is not None or selection_patience is not None)
    ):
        raise ValueError(
            "selection policy options require --select-best-checkpoint"
        )
    if selection_requested is None:
        selection = (
            None if checkpoint_config is None else checkpoint_config.selection
        )
    elif selection_requested:
        selection = CheckpointSelectionConfig(
            min_delta=_pick(selection_min_delta, 0.0),
            patience=_pick(selection_patience, 0),
        )
    else:
        selection = None

    return TrainConfig(
        lr=_pick(
            _optional_arg(args, "lr", float),
            DEFAULT_LR if checkpoint_config is None else checkpoint_config.lr,
        ),
        batch_size=_pick(
            _optional_arg(args, "batch_size", int),
            (
                DEFAULT_BATCH_SIZE
                if checkpoint_config is None
                else checkpoint_config.batch_size
            ),
        ),
        epochs=_pick(
            _optional_arg(args, "epochs", int),
            (
                DEFAULT_EPOCHS
                if checkpoint_config is None
                else checkpoint_config.epochs
            ),
        ),
        use_amp=_pick(
            _optional_arg(args, "use_amp", bool),
            False if checkpoint_config is None else checkpoint_config.use_amp,
        ),
        weight_decay=_pick(
            _optional_arg(args, "weight_decay", float),
            (
                DEFAULT_WEIGHT_DECAY
                if checkpoint_config is None
                else checkpoint_config.weight_decay
            ),
        ),
        selection=selection,
        seed=_pick(
            _optional_arg(args, "seed", int),
            (
                DEFAULT_SEED
                if checkpoint_config is None
                else checkpoint_config.seed
            ),
        ),
        deterministic=_pick(
            _optional_arg(args, "deterministic", bool),
            (
                DEFAULT_DETERMINISTIC
                if checkpoint_config is None
                else checkpoint_config.deterministic
            ),
        ),
    )


def objective_config_from_args(args: object) -> ObjectiveConfig:
    weights = _optional_float_tuple(args, "direct_loss_weights")
    return default_objective(
        DEFAULT_DIRECT_LOSS_WEIGHTS if weights is None else weights
    )


def _validate_checkpoint_model_overrides(
    args: object,
    checkpoint_config: ModelConfig,
) -> None:
    options = (
        ("seq_len", "seq-len", checkpoint_config.seq_len),
        ("hidden", "hidden", checkpoint_config.hidden),
        ("layers", "layers", checkpoint_config.layers),
        ("dropout", "dropout", checkpoint_config.dropout),
        ("nhead", "nhead", checkpoint_config.nhead),
        ("context_mode", "mode", checkpoint_config.context_mode),
    )
    for attribute, option, checkpoint_value in options:
        cli_value = _argument(args, attribute)
        if cli_value is None:
            continue
        if cli_value != checkpoint_value:
            raise ValueError(
                f"--{option}={cli_value} conflicts with checkpoint value "
                f"{checkpoint_value}"
            )


def _coerce_model_config(value: object) -> ModelConfig | None:
    if isinstance(value, ModelConfig) or value is None:
        return value
    return ModelConfig.from_dict(value)


def _coerce_train_config(value: object) -> TrainConfig | None:
    if isinstance(value, TrainConfig) or value is None:
        return value
    return TrainConfig.from_dict(value)


def _argument(args: object, name: str) -> object | None:
    return cast(object, getattr(args, name, None))


def _optional_arg(
    args: object,
    name: str,
    expected_type: type[T],
) -> T | None:
    value = _argument(args, name)
    if value is None:
        return None
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} has an invalid type")
    return value


def _optional_float_tuple(
    args: object,
    name: str,
) -> tuple[float, ...] | None:
    value = _argument(args, name)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} has an invalid type")
    items = cast(Sequence[object], value)
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in items
    ):
        raise ValueError(f"{name} must contain only numbers")
    return tuple(float(cast(int | float, item)) for item in items)
