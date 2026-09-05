from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from app.contracts.json_types import JsonObject, JsonValue

_SOURCE_KEYS = frozenset({"kind", "featureBlocks"})
_BLOCK_KEYS = frozenset({"position", "windowRows", "nativeRowWidth"})


def canonical_source_encoding(
    value: object,
    *,
    feature_dim: int,
) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError("sourceEncoding must be an object")
    untyped_source = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in untyped_source):
        raise ValueError("sourceEncoding field names must be strings")
    source = cast(Mapping[str, object], value)
    if frozenset(source) != _SOURCE_KEYS:
        raise ValueError(
            "sourceEncoding must contain exactly kind and featureBlocks"
        )
    if source.get("kind") != "indexedFeatureBlocks":
        raise ValueError(
            "sourceEncoding.kind must be indexedFeatureBlocks"
        )
    raw_blocks = source.get("featureBlocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("sourceEncoding.featureBlocks must be non-empty")
    feature_dim = _integer(feature_dim, "featureDim")
    if feature_dim <= 0:
        raise ValueError("featureDim must be positive")

    blocks: list[JsonValue] = []
    expected_position = 0
    for index, raw_block in enumerate(cast(list[object], raw_blocks)):
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"featureBlocks[{index}] must be an object")
        untyped_block = cast(Mapping[object, object], raw_block)
        if not all(isinstance(key, str) for key in untyped_block):
            raise ValueError(f"featureBlocks[{index}] field names must be strings")
        block = cast(Mapping[str, object], raw_block)
        if frozenset(block) != _BLOCK_KEYS:
            raise ValueError(
                f"featureBlocks[{index}] must contain exactly position, "
                "windowRows and nativeRowWidth"
            )
        position = _positive_or_zero(block.get("position"), "position")
        window_rows = _positive(block.get("windowRows"), "windowRows")
        native_row_width = _positive(
            block.get("nativeRowWidth"),
            "nativeRowWidth",
        )
        if position != expected_position:
            raise ValueError(
                f"featureBlocks[{index}].position must be "
                f"{expected_position}"
            )
        expected_position += window_rows * native_row_width
        blocks.append(cast(JsonValue, {
            "position": position,
            "windowRows": window_rows,
            "nativeRowWidth": native_row_width,
        }))

    if expected_position != feature_dim:
        raise ValueError(
            "sourceEncoding feature block widths must equal dataContract.featureDim"
        )
    return {
        "kind": "indexedFeatureBlocks",
        "featureBlocks": blocks,
    }


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
    return tuple(
        (
            cast(int, cast(Mapping[str, object], block)["position"]),
            cast(int, cast(Mapping[str, object], block)["windowRows"]),
            cast(int, cast(Mapping[str, object], block)["nativeRowWidth"]),
        )
        for block in blocks
    )


def _positive(value: object, name: str) -> int:
    parsed = _positive_or_zero(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_or_zero(value: object, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
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
