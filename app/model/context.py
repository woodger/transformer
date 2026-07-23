import torch

CONTEXT_MODES = ("strict", "relaxed")


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
    x: torch.Tensor,
    context_mode: str = "relaxed",
) -> torch.Tensor:
    context_mode = validate_context_mode(context_mode)
    missing = torch.isnan(x)
    if context_mode == "strict":
        return missing.any(dim=-1)
    return missing.all(dim=-1)


def context_token_ratios(
    x: torch.Tensor,
    context_mode: str = "relaxed",
) -> dict[str, float]:
    missing = torch.isnan(x)
    masked = context_key_padding_mask(x, context_mode)
    empty = missing.all(dim=-1)
    partial = missing.any(dim=-1) & ~empty
    complete = ~missing.any(dim=-1)

    return {
        "masked_token_ratio": float(masked.float().mean()),
        "complete_token_ratio": float(complete.float().mean()),
        "partial_token_ratio": float(partial.float().mean()),
        "empty_token_ratio": float(empty.float().mean()),
    }


def prepare_context_input(
    x: torch.Tensor,
    context_mode: str = "relaxed",
) -> tuple[torch.Tensor, torch.Tensor]:
    context_mode = validate_context_mode(context_mode)
    missing = torch.isnan(x)
    key_padding_mask = context_key_padding_mask(x, context_mode)

    # PyTorch attention can produce non-finite outputs when every token in a
    # sequence is masked. Keep one placeholder token attendable; NaN values
    # are filled below and relaxed mode still retains its missing flags.
    all_missing_rows = key_padding_mask.all(dim=1)
    if bool(all_missing_rows.any()):
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_missing_rows, 0] = False

    values = torch.nan_to_num(x, nan=0.0)
    if context_mode == "relaxed":
        values = torch.cat([values, missing.to(dtype=values.dtype)], dim=-1)

    return values, key_padding_mask
