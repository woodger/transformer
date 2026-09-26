from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from app.contracts.json_types import JsonObject, JsonValue

_LAYOUT_KEYS = frozenset({"featureBlocks"})
_BLOCK_KEYS = frozenset({"windowRows", "nativeRowWidth"})


def canonical_source_encoding(
    value: object,
    *,
    feature_dim: int,
) -> JsonObject:
    """Проверить единственную схему входных блоков v22 и вывести позиции.

    `indexedFeatureBlocks` привязан к версии. Его прежнее поле-различитель и
    накопленные позиции были избыточными входными полями.
    """

    if not isinstance(value, Mapping):
        raise ValueError("inputLayout must be an object")
    untyped_layout = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in untyped_layout):
        raise ValueError("inputLayout field names must be strings")
    layout = cast(Mapping[str, object], value)
    if frozenset(layout) != _LAYOUT_KEYS:
        raise ValueError("inputLayout must contain exactly featureBlocks")
    raw_blocks = layout.get("featureBlocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("inputLayout.featureBlocks must be non-empty")
    feature_dim = _integer(feature_dim, "featureDim")
    if feature_dim <= 0:
        raise ValueError("featureDim must be positive")

    blocks: list[JsonValue] = []
    width = 0
    for index, raw_block in enumerate(cast(list[object], raw_blocks)):
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"featureBlocks[{index}] must be an object")
        untyped_block = cast(Mapping[object, object], raw_block)
        if not all(isinstance(key, str) for key in untyped_block):
            raise ValueError(f"featureBlocks[{index}] field names must be strings")
        block = cast(Mapping[str, object], raw_block)
        if frozenset(block) != _BLOCK_KEYS:
            raise ValueError(
                f"featureBlocks[{index}] must contain exactly windowRows "
                "and nativeRowWidth"
            )
        window_rows = _positive(block.get("windowRows"), "windowRows")
        native_row_width = _positive(
            block.get("nativeRowWidth"),
            "nativeRowWidth",
        )
        width += window_rows * native_row_width
        blocks.append(cast(JsonValue, {
            "windowRows": window_rows,
            "nativeRowWidth": native_row_width,
        }))

    if width != feature_dim:
        raise ValueError(
            "inputLayout feature block widths must equal tensorGeometry.featureDim"
        )
    return {"featureBlocks": blocks}


def feature_block_dimensions(
    source_encoding: Mapping[str, object],
    *,
    feature_dim: int,
) -> tuple[tuple[int, int, int], ...]:
    canonical = canonical_source_encoding(
        source_encoding,
        feature_dim=feature_dim,
    )
    blocks = cast(list[object], canonical["featureBlocks"])
    position = 0
    dimensions: list[tuple[int, int, int]] = []
    for block in blocks:
        values = cast(Mapping[str, object], block)
        window_rows = cast(int, values["windowRows"])
        native_row_width = cast(int, values["nativeRowWidth"])
        dimensions.append((position, window_rows, native_row_width))
        position += window_rows * native_row_width
    return tuple(dimensions)


def _positive(value: object, name: str) -> int:
    parsed = _integer(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


__all__ = ["canonical_source_encoding", "feature_block_dimensions"]
