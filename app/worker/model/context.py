from dataclasses import dataclass

import torch

CONTEXT_MODES = ("strict", "relaxed")


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """Model features paired with their Transformer padding mask."""

    features: torch.Tensor
    key_padding_mask: torch.Tensor


def _validate_features(features: torch.Tensor) -> None:
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, sequence, features]")
    if features.shape[1] == 0 or features.shape[2] == 0:
        raise ValueError("sequence and feature dimensions must be positive")
    if features.dtype != torch.float32:
        raise ValueError("features must use float32")


def validate_context_mode(context_mode: str) -> str:
    if context_mode not in CONTEXT_MODES:
        choices = ", ".join(CONTEXT_MODES)
        raise ValueError(f"context_mode must be one of: {choices}")
    return context_mode


def context_input_dim(input_dim: int, context_mode: str) -> int:
    context_mode = validate_context_mode(context_mode)
    if context_mode == "relaxed":
        return input_dim * 2
    return input_dim


def context_key_padding_mask(
    features: torch.Tensor,
    context_mode: str = "relaxed",
) -> torch.Tensor:
    """Return bool [batch, sequence], where True marks an ignored token."""

    _validate_features(features)
    context_mode = validate_context_mode(context_mode)
    missing = torch.isnan(features)
    return _key_padding_mask(missing, context_mode)


def _key_padding_mask(
    missing: torch.Tensor,
    context_mode: str,
) -> torch.Tensor:
    if context_mode == "strict":
        return missing.any(dim=-1)
    return missing.all(dim=-1)


def context_missingness_ratios(
    features: torch.Tensor,
    context_mode: str = "relaxed",
) -> dict[str, float]:
    """Measure NaN and token ratios for float32 [batch, sequence, features]."""

    _validate_features(features)
    context_mode = validate_context_mode(context_mode)
    missing = torch.isnan(features)
    any_missing = missing.any(dim=-1)
    empty = missing.all(dim=-1)
    masked = any_missing if context_mode == "strict" else empty
    partial = any_missing & ~empty

    return {
        "nan_ratio": float(missing.float().mean()),
        "masked_token_ratio": float(masked.float().mean()),
        "complete_token_ratio": float((~any_missing).float().mean()),
        "partial_token_ratio": float(partial.float().mean()),
        "empty_token_ratio": float(empty.float().mean()),
    }


def context_token_ratios(
    features: torch.Tensor,
    context_mode: str = "relaxed",
) -> dict[str, float]:
    ratios = context_missingness_ratios(features, context_mode)
    return {
        key: value
        for key, value in ratios.items()
        if key != "nan_ratio"
    }


def prepare_context_input(
    features: torch.Tensor,
    context_mode: str = "relaxed",
) -> PreparedContext:
    """Prepare float32 features and bool padding mask for TransformerEncoder.

    Args:
        features: Tensor [batch, sequence, features]. NaN denotes a missing value.
        context_mode: ``strict`` masks partially missing tokens; ``relaxed``
            masks only fully missing tokens and appends per-feature missing flags.

    Returns:
        Prepared float32 features and bool mask [batch, sequence]. In the mask,
        ``True`` means that attention must ignore the token.
    """

    _validate_features(features)
    context_mode = validate_context_mode(context_mode)
    missing = torch.isnan(features)
    key_padding_mask = _key_padding_mask(missing, context_mode)

    # Механизм внимания PyTorch может выдавать не конечные значения, когда каждый
    # токен последовательности замаскирован. Оставляем один служебный токен
    # доступным для внимания; значения NaN заполняются ниже, а мягкий режим
    # сохраняет свои флаги отсутствия.
    all_missing_rows = key_padding_mask.all(dim=1)
    key_padding_mask[:, 0] &= ~all_missing_rows

    values = torch.nan_to_num(features, nan=0.0)
    if context_mode == "relaxed":
        values = torch.cat([values, missing.to(dtype=values.dtype)], dim=-1)

    return PreparedContext(
        features=values,
        key_padding_mask=key_padding_mask,
    )
