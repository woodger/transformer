from app.contracts.json_types import JsonObject
from app.contracts.worker.v15.checkpoint_selection import (
    DEFAULT_PATIENCE,
    CheckpointSelectionConfig,
)
from app.contracts.worker.v15.model_config import (
    DEFAULT_CONTEXT_MODE,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_NHEAD,
    ModelConfig,
)
from app.contracts.worker.v15.train_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DETERMINISTIC,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    TrainConfig,
)


def model_config_to_manifest(config: ModelConfig) -> JsonObject:
    return config.to_manifest()


def train_config_to_manifest(config: TrainConfig) -> JsonObject:
    return config.to_manifest()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONTEXT_MODE",
    "DEFAULT_DETERMINISTIC",
    "DEFAULT_DROPOUT",
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN",
    "DEFAULT_LAYERS",
    "DEFAULT_LR",
    "DEFAULT_NHEAD",
    "DEFAULT_PATIENCE",
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT_DECAY",
    "CheckpointSelectionConfig",
    "ModelConfig",
    "TrainConfig",
    "model_config_to_manifest",
    "train_config_to_manifest",
]
