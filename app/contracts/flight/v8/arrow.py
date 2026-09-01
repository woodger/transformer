from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app.contracts.ml import TARGET_IDENTITIES, canonical_targets


class _NumpyConvertible(Protocol):
    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray: ...


class _FlatPredictionArray(Protocol):
    def is_null(self) -> _NumpyConvertible: ...

    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray: ...


class _PredictionArray(Protocol):
    def is_null(self) -> _NumpyConvertible: ...

    def flatten(self) -> _FlatPredictionArray: ...


class _PredictionBatch(Protocol):
    @property
    def num_rows(self) -> int: ...

    def column(self, index: int) -> _PredictionArray: ...


class _PredictionReader(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

    @property
    def num_record_batches(self) -> int: ...

    def get_batch(self, index: int) -> _PredictionBatch: ...


@dataclass(frozen=True, slots=True)
class PredictionArrowStats:
    rows: int
    batches: int
    schema_fingerprint: str


def validate_prediction_file(
    path: str,
    prediction_column: str,
    expected_rows: int,
    targets: Sequence[str],
) -> PredictionArrowStats:
    selected = canonical_targets(targets)
    width = len(selected)
    rows = 0
    batches = 0
    with pa.memory_map(path, "r") as source:
        reader = cast(_PredictionReader, ipc.RecordBatchFileReader(source))
        schema = reader.schema
        _validate_prediction_schema(schema, prediction_column, selected)
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
                    f"{rows + first_invalid // width + 1}"
                )
            validate_target_space_values(values.reshape(-1, width), selected)
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
        pa.field(
            field.name,
            _type_without_metadata(field.type),
            nullable=field.nullable,
        )
        for field in schema
    ])
    return hashlib.sha256(canonical.serialize().to_pybytes()).hexdigest()


def canonical_input_schema(
    operation: str,
    source_width: int,
    targets: Sequence[str] = TARGET_IDENTITIES,
) -> pa.Schema:
    if operation not in ("fit", "predict"):
        raise ValueError("operation must be fit or predict")
    if type(source_width) is not int or source_width <= 0:
        raise ValueError("source_width must be a positive integer")
    fields = [
        pa.field(
            "src",
            _fixed_size_float32(source_width),
            nullable=False,
        ),
    ]
    if operation == "fit":
        selected = canonical_targets(targets)
        fields.append(
            pa.field(
                "tgt",
                _fixed_size_float32(len(selected)),
                nullable=False,
            )
        )
    return pa.schema(fields)


def canonical_prediction_schema(
    prediction_column: str,
    targets: Sequence[str] = TARGET_IDENTITIES,
) -> pa.Schema:
    if type(prediction_column) is not str or not prediction_column:
        raise ValueError("prediction_column must be a non-empty string")
    selected = canonical_targets(targets)
    return pa.schema([
        pa.field(
            prediction_column,
            _fixed_size_float32(len(selected)),
            nullable=False,
        ),
    ])


def _validate_prediction_schema(
    schema: pa.Schema,
    column: str,
    targets: tuple[str, ...],
) -> None:
    expected = canonical_prediction_schema(column, targets)
    if schema.names != expected.names:
        raise ValueError(
            f"prediction output must contain exactly column {column!r}"
        )
    field = schema.field(column)
    if field.type != expected.field(column).type:
        raise ValueError(
            "prediction output must use "
            f"FixedSizeList<float32>[{len(targets)}]"
        )
    if field.nullable:
        raise ValueError("prediction output column must be non-nullable")


def _fixed_size_float32(width: int) -> pa.FixedSizeListType:
    return pa.list_(
        pa.field("item", pa.float32(), nullable=True),
        width,
    )


def _type_without_metadata(value_type: pa.DataType) -> pa.DataType:
    if not pa.types.is_fixed_size_list(value_type):
        return value_type
    value_field = value_type.value_field
    return pa.list_(
        pa.field(
            value_field.name,
            _type_without_metadata(value_field.type),
            nullable=value_field.nullable,
        ),
        value_type.list_size,
    )


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


def validate_target_space_values(
    values: np.ndarray,
    targets: Sequence[str] = TARGET_IDENTITIES,
) -> None:
    selected = canonical_targets(targets)
    if values.ndim != 2 or values.shape[1] != len(selected):
        raise ValueError(
            f"target-space values must have width {len(selected)}"
        )
    if values.shape[0] == 0:
        return
    nonfinite_mask = cast(
        np.ndarray,
        ~np.isfinite(values).all(axis=1),
    )
    nonfinite_row = _first_true(nonfinite_mask)
    if nonfinite_row is not None:
        raise ValueError(
            f"target-space values must be finite at row {nonfinite_row + 1}"
        )

    for index, target in enumerate(selected):
        lower = -1.0 if target == "MeanReturn" else 0.0
        invalid = (values[:, index] < lower) | (values[:, index] > 1.0)
        invalid_row = _first_true(invalid)
        if invalid_row is not None:
            raise ValueError(
                f"target-space {target} is outside [{lower:g}, 1] at row "
                f"{invalid_row + 1}"
            )


__all__ = [
    "PredictionArrowStats",
    "canonical_input_schema",
    "canonical_prediction_schema",
    "schema_fingerprint",
    "validate_prediction_file",
    "validate_target_space_values",
]
