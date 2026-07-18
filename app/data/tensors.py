import torch


def reshape_source(X_cpu: torch.Tensor, seq_len: int) -> torch.Tensor:
    if seq_len is None or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")

    total_feat = X_cpu.shape[1]
    if total_feat % seq_len != 0:
        raise ValueError("Feature dim not divisible by seq_len")

    feat_dim = total_feat // seq_len
    return X_cpu.view(X_cpu.size(0), seq_len, feat_dim)


def validate_feature_dim(X_cpu: torch.Tensor, expected_feat_dim: int | None) -> int:
    feat_dim = X_cpu.shape[2]
    if expected_feat_dim is not None and feat_dim != expected_feat_dim:
        raise ValueError(
            f"Feature dim changed across frames: expected {expected_feat_dim}, got {feat_dim}"
        )
    return feat_dim


def validate_checkpoint_feature_dim(
    X_cpu: torch.Tensor,
    checkpoint_feature_dim: int | None,
) -> int:
    feature_dim = X_cpu.shape[2]
    if checkpoint_feature_dim is not None and feature_dim != checkpoint_feature_dim:
        raise ValueError(
            f"Feature dim {feature_dim} does not match checkpoint feature_dim "
            f"{checkpoint_feature_dim}"
        )
    return feature_dim


def validate_target_dim(Y_cpu: torch.Tensor, expected_target_dim: int | None) -> int:
    target_dim = Y_cpu.shape[1]
    if expected_target_dim is not None and target_dim != expected_target_dim:
        raise ValueError(
            f"Target dim changed across frames: expected {expected_target_dim}, got {target_dim}"
        )
    return target_dim
