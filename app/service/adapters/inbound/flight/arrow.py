from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from app.contracts.flight.v23.arrow import (
    canonical_input_schema,
    schema_fingerprint,
    validate_prediction_file as validate_contract_prediction_file,
    validate_target_values,
)
from app.contracts.flight.v23.source_encoding import feature_block_dimensions
from app.contracts.flight.v23.target_value_error import TargetValueError
from app.contracts.json_types import JsonObject
from app.service.adapters.inbound.flight.errors import invalid, resource_exhausted
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode

_POSTGRES_BIGINT_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class ArrowStats:
    rows: int
    batches: int
    schema_fingerprint: str
    chunks: int = 0
    native_rows: tuple[int, ...] = ()
    first_range_ordinal: int | None = None
    first_example_offset: int | None = None
    last_range_ordinal: int | None = None
    next_example_offset: int | None = None


class InputBatchValidator:
    """Проверить один компактный пакет DoPut без развёртывания логических тензоров."""

    def __init__(
        self,
        operation: str,
        schema: pa.Schema,
        *,
        source_encoding: Mapping[str, object],
        target_contract: JsonObject,
        binary_target_indices: Sequence[int] = (),
        seq_len: int,
        expected_feature_dim: int,
        max_batch_bytes: int,
        max_payload_bytes: int,
        max_rows: int,
    ) -> None:
        if operation not in ("fit", "predict"):
            raise invalid("unsupported input operation")
        self.operation = operation
        self.target_contract = target_contract
        self.binary_target_indices = tuple(binary_target_indices)
        self.schema = schema
        self.seq_len = seq_len
        self.expected_feature_dim = expected_feature_dim
        self.blocks = feature_block_dimensions(
            source_encoding,
            feature_dim=expected_feature_dim,
        )
        self.max_batch_bytes = max_batch_bytes
        self.max_payload_bytes = max_payload_bytes
        self.max_rows = max_rows
        self.rows = 0
        self.chunks = 0
        self.batches = 0
        self.logical_bytes = 0
        self.native_rows = [0] * len(self.blocks)
        self.first_range_ordinal: int | None = None
        self.first_example_offset: int | None = None
        self.last_range_ordinal: int | None = None
        self.next_example_offset: int | None = None

        expected_schema = canonical_input_schema(
            operation,
            source_encoding,
            seq_len,
            expected_feature_dim,
            self.target_contract,
        )
        if not schema.equals(expected_schema, check_metadata=False):
            raise invalid(
                f"{operation} input differs from the canonical physical schema"
            )

    @property
    def fingerprint(self) -> str:
        return schema_fingerprint(self.schema)

    def validate_batch(self, batch: pa.RecordBatch) -> None:
        if not batch.schema.equals(self.schema, check_metadata=False):
            raise invalid("RecordBatch schema differs from the DoPut schema")
        if batch.nbytes > self.max_batch_bytes:
            raise resource_exhausted(
                f"RecordBatch size {batch.nbytes} exceeds limit {self.max_batch_bytes}"
            )
        if self.logical_bytes + batch.nbytes > self.max_payload_bytes:
            raise resource_exhausted(
                f"payload data size exceeds limit {self.max_payload_bytes}"
            )

        try:
            logical_rows, native_rows = self._validate_compact_batch(batch)
        except ValueError as exc:
            raise invalid(str(exc)) from exc
        if self.rows + logical_rows > self.max_rows:
            raise resource_exhausted(
                f"payload logical row count exceeds limit {self.max_rows}"
            )

        self.rows += logical_rows
        self.chunks += batch.num_rows
        self.batches += 1
        self.logical_bytes += batch.nbytes
        for index, count in enumerate(native_rows):
            self.native_rows[index] += count

    def stats(self) -> ArrowStats:
        return ArrowStats(
            rows=self.rows,
            batches=self.batches,
            schema_fingerprint=self.fingerprint,
            chunks=self.chunks,
            native_rows=tuple(self.native_rows),
            first_range_ordinal=self.first_range_ordinal,
            first_example_offset=self.first_example_offset,
            last_range_ordinal=self.last_range_ordinal,
            next_example_offset=self.next_example_offset,
        )

    def _validate_compact_batch(
        self,
        batch: pa.RecordBatch,
    ) -> tuple[int, tuple[int, ...]]:
        range_ordinals = batch.column(
            self.schema.get_field_index("rangeOrdinal")
        )
        example_offsets = batch.column(
            self.schema.get_field_index("exampleOffset")
        )
        features: pa.StructArray = batch.column(
            self.schema.get_field_index("features")
        )
        _require_no_nulls(range_ordinals, "rangeOrdinal")
        _require_no_nulls(example_offsets, "exampleOffset")
        _require_no_nulls(features, "features")
        ranges = range_ordinals.to_numpy(zero_copy_only=False)
        starts = example_offsets.to_numpy(zero_copy_only=False)

        logical_lengths: npt.NDArray[np.signedinteger] | None = None
        native_totals: list[int] = []
        for index, (_, window_rows, _) in enumerate(self.blocks):
            block: pa.StructArray = features.field(index)
            _require_no_nulls(block, f"features.b{index}")
            native: pa.ListArray = block.field("nativeRows")
            offsets: pa.ListArray = block.field("observationOffsets")
            _require_no_nulls(native, f"features.b{index}.nativeRows")
            _require_no_nulls(
                offsets,
                f"features.b{index}.observationOffsets",
            )
            native_lengths = np.diff(
                native.offsets.to_numpy(zero_copy_only=False)
            )
            block_lengths = np.diff(
                offsets.offsets.to_numpy(zero_copy_only=False)
            )
            if logical_lengths is None:
                logical_lengths = block_lengths
            elif not np.array_equal(logical_lengths, block_lengths):
                raise ValueError(
                    "feature block observation counts differ within a range chunk"
                )

            native_values: pa.FixedSizeListArray = native.flatten()
            offset_values: pa.FixedSizeListArray = offsets.flatten()
            _validate_native_values(native_values, index)
            _require_no_nulls(
                offset_values,
                f"features.b{index}.observationOffsets row",
            )
            flat_offsets = offset_values.flatten()
            _require_no_nulls(
                flat_offsets,
                f"features.b{index}.observationOffsets value",
            )
            offset_matrix = flat_offsets.to_numpy(
                zero_copy_only=False
            ).reshape(-1, self.seq_len)
            logical_cursor = 0
            for chunk_index, (native_count, logical_count) in enumerate(
                zip(native_lengths, block_lengths, strict=True)
            ):
                next_cursor = logical_cursor + int(logical_count)
                local_offsets = offset_matrix[logical_cursor:next_cursor]
                if local_offsets.size:
                    largest = int(local_offsets.max())
                    if largest + window_rows > int(native_count):
                        raise ValueError(
                            f"features.b{index} observation offset is outside "
                            f"range chunk {chunk_index + 1}"
                        )
                logical_cursor = next_cursor
            native_totals.append(int(native_lengths.sum()))

        if logical_lengths is None:
            raise ValueError("sourceEncoding must contain a feature block")
        if batch.num_rows and np.any(logical_lengths <= 0):
            raise ValueError("range chunks must contain at least one logical row")
        logical_rows = int(logical_lengths.sum())

        if self.operation == "fit":
            targets: pa.ListArray = batch.column(
                self.schema.get_field_index("tgt")
            )
            _require_no_nulls(targets, "tgt")
            target_lengths = np.diff(
                targets.offsets.to_numpy(zero_copy_only=False)
            )
            if not np.array_equal(logical_lengths, target_lengths):
                raise ValueError(
                    "tgt count differs from feature observations within a range chunk"
                )
            target_rows: pa.FixedSizeListArray = targets.flatten()
            _require_no_nulls(target_rows, "tgt row")
            target_values = target_rows.flatten()
            _require_no_nulls(target_values, "tgt value")
            values = target_values.to_numpy(zero_copy_only=False).reshape(
                -1,
                len(cast(Sequence[object], self.target_contract["slots"])),
            )
            try:
                validate_target_values(
                    values,
                    self.target_contract,
                    logical_row_offset=self.rows,
                    binary_target_indices=self.binary_target_indices,
                )
            except TargetValueError as exc:
                detail: JsonObject = {
                    "code": "INVALID_ARGUMENT",
                    "reason": "TARGET_VALUE_INVALID",
                    "targetIdentity": exc.target_identity,
                    "targetIndex": exc.target_index,
                    "logicalRow": exc.logical_row,
                    "message": str(exc),
                }
                if exc.expected_domain is not None:
                    detail["expectedDomain"] = exc.expected_domain
                raise ServiceError(
                    ErrorCode.INVALID_ARGUMENT,
                    str(exc),
                    detail=detail,
                ) from exc

        for index, (raw_range, raw_start, logical_count) in enumerate(
            zip(ranges, starts, logical_lengths, strict=True)
        ):
            range_ordinal = int(raw_range)
            example_offset = int(raw_start)
            next_offset = example_offset + int(logical_count)
            if next_offset > _POSTGRES_BIGINT_MAX:
                raise ValueError(
                    f"range chunk {index + 1} exceeds the supported example offset"
                )
            self._accept_boundary(range_ordinal, example_offset, next_offset)

        return logical_rows, tuple(native_totals)

    def _accept_boundary(
        self,
        range_ordinal: int,
        example_offset: int,
        next_offset: int,
    ) -> None:
        if self.last_range_ordinal is not None:
            if range_ordinal == self.last_range_ordinal:
                if example_offset != self.next_example_offset:
                    raise ValueError(
                        "exampleOffset is not contiguous within rangeOrdinal"
                    )
            elif (
                range_ordinal != self.last_range_ordinal + 1
                or example_offset != 0
            ):
                raise ValueError(
                    "rangeOrdinal must be dense and each new range must start at zero"
                )
        if self.first_range_ordinal is None:
            self.first_range_ordinal = range_ordinal
            self.first_example_offset = example_offset
        self.last_range_ordinal = range_ordinal
        self.next_example_offset = next_offset


def validate_prediction_file(
    path: str,
    prediction_column: str,
    expected_rows: int,
    target_contract: JsonObject,
) -> ArrowStats:
    try:
        stats = validate_contract_prediction_file(
            path,
            prediction_column,
            expected_rows,
            target_contract,
        )
    except ValueError as exc:
        raise invalid(str(exc)) from exc
    return ArrowStats(
        rows=stats.rows,
        batches=stats.batches,
        schema_fingerprint=stats.schema_fingerprint,
    )


def _validate_native_values(
    rows: pa.FixedSizeListArray,
    block_index: int,
) -> None:
    _require_no_nulls(rows, f"features.b{block_index}.nativeRows row")
    values = rows.flatten()
    _require_no_nulls(values, f"features.b{block_index}.nativeRows value")
    array = values.to_numpy(zero_copy_only=False)
    if np.isinf(array).any():
        raise ValueError(
            f"features.b{block_index}.nativeRows contains an infinite value"
        )


def _require_no_nulls(array: pa.Array, name: str) -> None:
    if array.null_count:
        raise ValueError(f"Arrow field '{name}' must not contain nulls")


__all__ = [
    "ArrowStats",
    "InputBatchValidator",
    "schema_fingerprint",
    "validate_prediction_file",
]
