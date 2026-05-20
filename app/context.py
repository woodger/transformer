import torch


def context_key_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isnan(x).all(dim=-1)


def context_valid_token_ratio(x: torch.Tensor) -> float:
    return float((~context_key_padding_mask(x)).float().mean())


def prepare_context_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    missing = torch.isnan(x)
    key_padding_mask = missing.all(dim=-1)

    # PyTorch attention can produce non-finite outputs when every token in a
    # sequence is masked. Keep one zero-valued placeholder token attendable.
    all_missing_rows = key_padding_mask.all(dim=1)
    if bool(all_missing_rows.any()):
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_missing_rows, 0] = False

    values = torch.nan_to_num(x, nan=0.0)
    values = torch.cat([values, missing.to(dtype=values.dtype)], dim=-1)

    return values, key_padding_mask
