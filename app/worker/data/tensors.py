from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Данные fit Worker с одной выровненной осью строк.

    ``features`` имеет float32 ``[rows, source_width]`` на границе Arrow и
    float32 ``[batch, sequence, features]`` после преобразования источника.
    ``targets`` имеет float32 ``[rows, selected_targets]`` в публичном порядке
    target.
    """

    features: torch.Tensor
    targets: torch.Tensor


def reshape_source(
    features: torch.Tensor,
    seq_len: int | None,
) -> torch.Tensor:
    """Преобразовать плоские признаки CPU [rows, seq*features] в тензор ранга 3."""

    if seq_len is None or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if features.ndim != 2:
        raise ValueError("source features must have shape [rows, flattened_features]")

    total_feat = features.shape[1]
    if total_feat % seq_len != 0:
        raise ValueError("Feature dim not divisible by seq_len")

    feat_dim = total_feat // seq_len
    return features.view(features.size(0), seq_len, feat_dim)


def validate_feature_dim(
    features: torch.Tensor,
    expected_feat_dim: int | None,
) -> int:
    """Проверить ось ``features`` у [rows, sequence, features]."""

    if features.ndim != 3:
        raise ValueError("source features must have shape [rows, sequence, features]")
    feat_dim = features.shape[2]
    if expected_feat_dim is not None and feat_dim != expected_feat_dim:
        raise ValueError(
            f"Feature dim changed across frames: expected {expected_feat_dim}, got {feat_dim}"
        )
    return feat_dim


def validate_checkpoint_feature_dim(
    features: torch.Tensor,
    checkpoint_feature_dim: int | None,
) -> int:
    """Сопоставить [rows, sequence, features] с метаданными контрольной точки."""

    if checkpoint_feature_dim is None:
        raise ValueError("Checkpoint feature_dim is unavailable")
    if features.ndim != 3:
        raise ValueError("source features must have shape [rows, sequence, features]")
    feature_dim = features.shape[2]
    if feature_dim != checkpoint_feature_dim:
        raise ValueError(
            f"Feature dim {feature_dim} does not match checkpoint feature_dim "
            f"{checkpoint_feature_dim}"
        )
    return feature_dim


def validate_target_dim(
    targets: torch.Tensor,
    expected_target_dim: int | None,
) -> int:
    """Проверить ширину тензора target [rows, targets]."""

    if targets.ndim != 2:
        raise ValueError("targets must have shape [rows, targets]")
    target_dim = targets.shape[1]
    if expected_target_dim is not None and target_dim != expected_target_dim:
        raise ValueError(
            f"Target dim changed across frames: expected {expected_target_dim}, got {target_dim}"
        )
    return target_dim
