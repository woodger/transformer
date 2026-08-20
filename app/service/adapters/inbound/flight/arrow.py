from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from app.contracts.flight.v5.arrow import (
    TARGET_WIDTH,
    canonical_input_schema,
    schema_fingerprint,
    validate_prediction_file as validate_contract_prediction_file,
    validate_target_space_values,
)
from app.service.adapters.inbound.flight.errors import invalid, resource_exhausted

_FLOAT32_MAX = float(np.finfo(np.float32).max)


class _FixedSizeListType(Protocol):
    list_size: int
    value_type: object


class _ArrowArray(Protocol):
    type: object
    offsets: "_ArrowArray"

    def __len__(self) -> int: ...

    def is_null(self) -> "_ArrowArray": ...

    def flatten(self) -> "_ArrowArray": ...

    def to_numpy(
        self,
        *,
        zero_copy_only: bool,
    ) -> npt.NDArray[np.generic]: ...


@dataclass(frozen=True, slots=True)
class ArrowStats:
    rows: int
    batches: int
    schema_fingerprint: str
    src_width: int | None = None
    tgt_width: int | None = None


class InputBatchValidator:
    """Validate and accumulate one DoPut payload before durable publication.

    Construction enforces the canonical physical schema; each batch then
    enforces the value, row, and byte limits for that same payload.
    """

    def __init__(
        self,
        operation: str,
        schema: pa.Schema,
        *,
        seq_len: int,
        expected_feature_dim: int | None,
        max_batch_bytes: int,
        max_payload_bytes: int,
        max_rows: int,
    ) -> None:
        if operation not in ("fit", "predict"):
            raise invalid("unsupported input operation")
        self.operation = operation
        self.schema = schema
        self.seq_len = seq_len
        self.expected_feature_dim = expected_feature_dim
        self.max_batch_bytes = max_batch_bytes
        self.max_payload_bytes = max_payload_bytes
        self.max_rows = max_rows
        self.rows = 0
        self.batches = 0
        self.logical_bytes = 0
        _validate_input_schema(schema, operation)
        source_width = _fixed_width(schema, "src")
        if source_width is None:
            raise invalid("src must use a FixedSizeList physical type")
        self.src_width = source_width
        self.tgt_width = _fixed_width(schema, "tgt") if operation == "fit" else None
        self._validate_source_width(self.src_width)
        if self.tgt_width is not None and self.tgt_width != TARGET_WIDTH:
            raise invalid(f"tgt list width must be {TARGET_WIDTH}")
        expected_schema = canonical_input_schema(operation, self.src_width)
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
        if self.rows + batch.num_rows > self.max_rows:
            raise resource_exhausted(
                f"payload row count exceeds limit {self.max_rows}"
            )

        table = pa.Table.from_batches([batch], schema=self.schema)
        try:
            _validate_arrow_table(table, require_target=self.operation == "fit")
        except ValueError as exc:
            raise invalid(str(exc)) from exc

        source_width = _observed_width(batch.column(self.schema.get_field_index("src")))
        target_width = None
        if self.operation == "fit":
            target_width = _observed_width(
                batch.column(self.schema.get_field_index("tgt"))
            )
        self.src_width = _merge_width("src", self.src_width, source_width)
        self.tgt_width = _merge_width("tgt", self.tgt_width, target_width)
        self._validate_source_width(self.src_width)
        if self.tgt_width is not None and self.tgt_width != TARGET_WIDTH:
            raise invalid(f"tgt list width must be {TARGET_WIDTH}")

        self.rows += batch.num_rows
        self.batches += 1
        self.logical_bytes += batch.nbytes

    def stats(self) -> ArrowStats:
        return ArrowStats(
            rows=self.rows,
            batches=self.batches,
            schema_fingerprint=self.fingerprint,
            src_width=self.src_width,
            tgt_width=self.tgt_width,
        )

    def _validate_source_width(self, width: int | None) -> None:
        if width is None:
            return
        if width <= 0:
            raise invalid("src list width must be positive")
        if width % self.seq_len != 0:
            raise invalid(
                f"src list width {width} is not divisible by seqLen {self.seq_len}"
            )
        feature_dim = width // self.seq_len
        if (
            self.expected_feature_dim is not None
            and feature_dim != self.expected_feature_dim
        ):
            raise invalid(
                f"input feature dimension {feature_dim} does not match model "
                f"feature dimension {self.expected_feature_dim}"
            )


def validate_prediction_file(
    path: str,
    prediction_column: str,
    expected_rows: int,
) -> ArrowStats:
    try:
        stats = validate_contract_prediction_file(
            path,
            prediction_column,
            expected_rows,
        )
    except ValueError as exc:
        raise invalid(str(exc)) from exc
    return ArrowStats(
        rows=stats.rows,
        batches=stats.batches,
        schema_fingerprint=stats.schema_fingerprint,
        src_width=TARGET_WIDTH,
    )


def _validate_input_schema(schema: pa.Schema, operation: str) -> None:
    expected = ["src", "tgt"] if operation == "fit" else ["src"]
    if schema.names != expected:
        raise invalid(
            f"{operation} input columns must be exactly {', '.join(expected)}"
        )
    for name in expected:
        field = schema.field(name)
        field_type = field.type
        if (
            not pa.types.is_fixed_size_list(field_type)
            or field_type.value_type != pa.float32()
        ):
            raise invalid(
                f"Arrow column '{name}' must be a FixedSizeList<float32> column"
            )
        if field.nullable:
            raise invalid(f"Arrow column '{name}' must be non-nullable")


def _validate_arrow_table(table: pa.Table, *, require_target: bool) -> None:
    _validate_list_column(table, "src", allow_nan=True)
    if require_target:
        target_values = _validate_list_column(
            table,
            "tgt",
            allow_nan=False,
            expected_width=TARGET_WIDTH,
        )
        _validate_target_values(target_values)


def _validate_list_column(
    table: pa.Table,
    name: str,
    *,
    allow_nan: bool,
    expected_width: int | None = None,
) -> np.ndarray:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Arrow table must contain '{name}' column")
    column_type = table.schema.field(column_index).type
    if (
        not pa.types.is_fixed_size_list(column_type)
        or column_type.value_type != pa.float32()
    ):
        raise ValueError(
            f"Arrow column '{name}' must be a FixedSizeList<float32> column"
        )

    width = cast(_FixedSizeListType, column_type).list_size
    chunks: list[tuple[int, int, npt.NDArray[np.float32]]] = []
    row_offset = 0
    for raw_chunk in table.column(column_index).chunks:
        chunk = cast(_ArrowArray, raw_chunk)
        if len(chunk) == 0:
            continue
        null_row = _first_true(chunk.is_null().to_numpy(zero_copy_only=False))
        if null_row is not None:
            raise ValueError(
                f"Arrow column '{name}' has null row at index "
                f"{row_offset + null_row + 1}"
            )
        flat = chunk.flatten()
        null_value = _first_true(flat.is_null().to_numpy(zero_copy_only=False))
        if null_value is not None:
            row_index, value_index = divmod(null_value, width)
            raise ValueError(
                f"Arrow column '{name}' has null element at row "
                f"{row_offset + row_index + 1}, position {value_index + 1}"
            )
        flat_values = cast(
            npt.NDArray[np.float32],
            flat.to_numpy(zero_copy_only=False),
        )
        invalid_values = (
            np.isinf(flat_values)
            if allow_nan
            else ~np.isfinite(flat_values)
        )
        invalid_value = _first_true(invalid_values)
        if invalid_value is not None:
            row_index, value_index = divmod(invalid_value, width)
            raise ValueError(
                f"Arrow column '{name}' has non-finite value at row "
                f"{row_offset + row_index + 1}, position {value_index + 1}"
            )
        outside_float32 = _first_true(np.abs(flat_values) > _FLOAT32_MAX)
        if outside_float32 is not None:
            row_index, value_index = divmod(outside_float32, width)
            raise ValueError(
                f"Arrow column '{name}' has value outside float32 range at "
                f"row {row_offset + row_index + 1}, position {value_index + 1}"
            )
        chunks.append((row_offset, len(chunk), flat_values))
        row_offset += len(chunk)

    if expected_width is not None and width != expected_width:
        raise ValueError(
            f"Arrow column '{name}' must have list length {expected_width}, got {width}"
        )
    if name == "src" and table.num_rows > 0 and width == 0:
        raise ValueError("Arrow column 'src' must have a positive list length")
    if table.num_rows == 0:
        return np.empty((0, 0), dtype=np.float32)
    values = np.empty((table.num_rows, width), dtype=np.float32)
    for offset, rows, flat_values in chunks:
        values[offset : offset + rows] = flat_values.reshape(rows, width)
    return values


def _validate_target_values(values: np.ndarray) -> None:
    validate_target_space_values(values)


def _fixed_width(schema: pa.Schema, name: str) -> int | None:
    index = schema.get_field_index(name)
    if index < 0:
        return None
    value_type = schema.field(index).type
    if pa.types.is_fixed_size_list(value_type):
        return cast(_FixedSizeListType, value_type).list_size
    return None


def _observed_width(value: object) -> int | None:
    array = cast(_ArrowArray, value)
    if len(array) == 0:
        return None
    if pa.types.is_fixed_size_list(array.type):
        return cast(_FixedSizeListType, array.type).list_size
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    return int(offsets[1] - offsets[0])


def _first_true(values: npt.ArrayLike) -> int | None:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


def _merge_width(name: str, current: int | None, observed: int | None) -> int | None:
    if observed is None:
        return current
    if current is not None and current != observed:
        raise invalid(
            f"Arrow column '{name}' has inconsistent list width across batches"
        )
    return observed
