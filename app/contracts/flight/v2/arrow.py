from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

_TARGET_WIDTH = 6


@dataclass(frozen=True, slots=True)
class PredictionArrowStats:
    rows: int
    batches: int
    schema_fingerprint: str


def validate_prediction_file(
    path: str,
    prediction_column: str,
    expected_rows: int,
) -> PredictionArrowStats:
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
                raise ValueError(
                    f"prediction has null row at index {rows + null_row + 1}"
                )
            lengths = np.diff(array.offsets.to_numpy(zero_copy_only=False))
            wrong_width = _first_true(lengths != _TARGET_WIDTH)
            if wrong_width is not None:
                raise ValueError(
                    f"prediction list width must be {_TARGET_WIDTH}, "
                    f"got {int(lengths[wrong_width])}"
                )
            flat = array.flatten()
            values = flat.to_numpy(zero_copy_only=False)
            invalid_value = np.logical_or(
                flat.is_null().to_numpy(zero_copy_only=False),
                ~np.isfinite(values),
            )
            first_invalid = _first_true(invalid_value)
            if first_invalid is not None:
                raise ValueError(
                    "prediction has non-finite or null value at row "
                    f"{rows + first_invalid // _TARGET_WIDTH + 1}"
                )
            rows += batch.num_rows
            batches += 1
    if rows != expected_rows:
        raise ValueError(
            f"prediction row count {rows} does not match input row count "
            f"{expected_rows}"
        )
    return PredictionArrowStats(
        rows=rows,
        batches=batches,
        schema_fingerprint=schema_fingerprint(schema),
    )


def schema_fingerprint(schema: pa.Schema) -> str:
    canonical = pa.schema([
        pa.field(field.name, field.type, nullable=field.nullable)
        for field in schema
    ])
    return hashlib.sha256(canonical.serialize().to_pybytes()).hexdigest()


def _validate_prediction_schema(schema: pa.Schema, column: str) -> None:
    if schema.names != [column]:
        raise ValueError(
            f"prediction output must contain exactly column {column!r}"
        )
    if schema.field(column).type != pa.list_(pa.float32()):
        raise ValueError("prediction output must use list<float32>")


def _first_true(values) -> int | None:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


__all__ = [
    "PredictionArrowStats",
    "schema_fingerprint",
    "validate_prediction_file",
]

