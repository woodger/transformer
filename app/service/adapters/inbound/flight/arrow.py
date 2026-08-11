import hashlib
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app.service.adapters.inbound.flight.errors import invalid, resource_exhausted

TARGET_WIDTH = 6
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class ArrowStats:
    rows: int
    batches: int
    schema_fingerprint: str
    src_width: int | None = None
    tgt_width: int | None = None


class InputBatchValidator:
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
    ):
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
        self.src_width = _fixed_width(schema, "src")
        self.tgt_width = _fixed_width(schema, "tgt") if operation == "fit" else None
        _validate_input_schema(schema, operation)
        self._validate_source_width(self.src_width)
        if self.tgt_width is not None and self.tgt_width != TARGET_WIDTH:
            raise invalid(f"tgt list width must be {TARGET_WIDTH}")

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
    rows = 0
    batches = 0
    with pa.memory_map(path, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        schema = reader.schema
        _validate_prediction_schema(schema, prediction_column)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            array = batch.column(0)
            null_row = _first_true(
                array.is_null().to_numpy(zero_copy_only=False)
            )
            if null_row is not None:
                raise invalid(
                    f"prediction has null row at index {rows + null_row + 1}"
                )

            flat = array.flatten()
            values = flat.to_numpy(zero_copy_only=False)
            invalid_value = np.logical_or(
                flat.is_null().to_numpy(zero_copy_only=False),
                ~np.isfinite(values),
            )
            first_invalid = _first_true(invalid_value)
            if first_invalid is not None:
                raise invalid(
                    "prediction has non-finite or null value at row "
                    f"{rows + first_invalid // TARGET_WIDTH + 1}"
                )
            rows += batch.num_rows
            batches += 1
    if rows != expected_rows:
        raise invalid(
            f"prediction row count {rows} does not match input row count "
            f"{expected_rows}"
        )
    return ArrowStats(
        rows=rows,
        batches=batches,
        schema_fingerprint=schema_fingerprint(schema),
        src_width=TARGET_WIDTH,
    )


def schema_fingerprint(schema: pa.Schema) -> str:
    canonical = pa.schema([
        pa.field(field.name, field.type, nullable=field.nullable)
        for field in schema
    ])
    return hashlib.sha256(canonical.serialize().to_pybytes()).hexdigest()


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

    width = column_type.list_size
    chunks = []
    row_offset = 0
    for chunk in table.column(column_index).chunks:
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
        flat_values = flat.to_numpy(zero_copy_only=False)
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

    if expected_width is not None and width is not None and width != expected_width:
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
    if values.shape[0] == 0:
        return
    invalid_volatility = values[:, 4] < 0.0
    invalid_probability = (values[:, 5] < 0.0) | (values[:, 5] > 1.0)
    invalid_row = _first_true(invalid_volatility | invalid_probability)
    if invalid_row is None:
        return
    if invalid_volatility[invalid_row]:
        raise ValueError(
            f"Arrow column 'tgt' has negative volatility at row {invalid_row + 1}"
        )
    raise ValueError(
        "Arrow column 'tgt' has hit probability outside [0, 1] "
        f"at row {invalid_row + 1}"
    )


def _validate_prediction_schema(schema: pa.Schema, column: str) -> None:
    if schema.names != [column]:
        raise invalid(f"prediction output must contain exactly column {column!r}")
    field = schema.field(column)
    field_type = field.type
    if field_type != pa.list_(pa.float32(), TARGET_WIDTH):
        raise invalid(
            f"prediction output must use FixedSizeList<float32>[{TARGET_WIDTH}]"
        )
    if field.nullable:
        raise invalid("prediction output column must be non-nullable")


def _fixed_width(schema: pa.Schema, name: str) -> int | None:
    index = schema.get_field_index(name)
    if index < 0:
        return None
    value_type = schema.field(index).type
    if pa.types.is_fixed_size_list(value_type):
        return value_type.list_size
    return None


def _observed_width(array) -> int | None:
    if len(array) == 0:
        return None
    if pa.types.is_fixed_size_list(array.type):
        return array.type.list_size
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    return int(offsets[1] - offsets[0])


def _first_true(values) -> int | None:
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
