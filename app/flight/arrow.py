from dataclasses import dataclass
import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app.data.arrow import TARGET_WIDTH, validate_arrow_table
from app.flight.errors import invalid, resource_exhausted


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
            validate_arrow_table(table, require_target=self.operation == "fit")
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
            values = batch.column(0).to_pylist()
            for row_index, row in enumerate(values, start=rows + 1):
                if row is None:
                    raise invalid(
                        f"prediction has null row at index {row_index}"
                    )
                if len(row) != TARGET_WIDTH:
                    raise invalid(
                        f"prediction list width must be {TARGET_WIDTH}, got {len(row)}"
                    )
                if any(value is None or not np.isfinite(value) for value in row):
                    raise invalid(
                        f"prediction has non-finite or null value at row {row_index}"
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
        field_type = schema.field(name).type
        if not _is_supported_list(field_type):
            raise invalid(
                f"Arrow column '{name}' must be a list<float32> or "
                "list<float64> column"
            )


def _validate_prediction_schema(schema: pa.Schema, column: str) -> None:
    if schema.names != [column]:
        raise invalid(f"prediction output must contain exactly column {column!r}")
    field_type = schema.field(column).type
    if field_type != pa.list_(pa.float32()):
        raise invalid("prediction output must use list<float32>")


def _is_supported_list(value_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(value_type)
        or pa.types.is_large_list(value_type)
        or pa.types.is_fixed_size_list(value_type)
    ) and value_type.value_type in (pa.float32(), pa.float64())


def _fixed_width(schema: pa.Schema, name: str) -> int | None:
    index = schema.get_field_index(name)
    if index < 0:
        return None
    value_type = schema.field(index).type
    if pa.types.is_fixed_size_list(value_type):
        return value_type.list_size
    return None


def _observed_width(array) -> int | None:
    for row in array.to_pylist():
        if row is not None:
            return len(row)
    return None


def _merge_width(name: str, current: int | None, observed: int | None) -> int | None:
    if observed is None:
        return current
    if current is not None and current != observed:
        raise invalid(
            f"Arrow column '{name}' has inconsistent list width across batches"
        )
    return observed
